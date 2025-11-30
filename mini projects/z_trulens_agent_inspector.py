"""
Instrument an agent (tool-using LLM) to evaluate each plan/tool call and find failure modes
using TruLens. This file provides a reusable AgentInspector that wraps an agent's execution
and records per-step traces, feedbacks, and aggregated metrics.
"""

from __future__ import annotations

import json
import time
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple

from trulens.apps.app import TruApp
from trulens_eval import Feedback

# -----------------------------
# Types & small agent example
# -----------------------------

ToolOutput = Dict[str, Any]
ToolFunc = Callable[[str, Dict[str, Any]], ToolOutput]


class Tool:
    """Simple tool wrapper for demonstration.

    A real agent might call external APIs, run code, query a DB, etc. Each tool
    should return a dict with at least a `text` field and optionally `meta`.
    """

    def __init__(self, name: str, func: ToolFunc):
        self.name = name
        self.func = func

    def run(self, arg_text: str, kwargs: Optional[Dict[str, Any]] = None) -> ToolOutput:
        kwargs = kwargs or {}
        return self.func(arg_text, kwargs)


class MockAgent:
    """A tiny example agent that alternates planning and tool calls.

    The `run_episode` method yields a sequence of step dicts so the Inspector can
    record them. Each step is either a 'plan' step or a 'tool' step.
    """

    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools

    def run_episode(self, user_input: str):
        # Example plan 1
        plan_1 = {
            "type": "plan",
            "plan_text": f"I will call search to look up: {user_input}",
            "timestamp": time.time(),
        }
        yield plan_1

        # Example tool call
        tool_call = {
            "type": "tool_call",
            "tool_name": "search",
            "tool_args": {"q": user_input},
            "timestamp": time.time(),
        }
        # Execute tool
        tool = self.tools[tool_call["tool_name"]]
        out: Any = tool.run(user_input)
        if not out:
            raise ValueError("Tool execution failed: `out` is None.")
        tool_call["tool_output"] = out
        tool_call["timestamp_done"] = time.time()
        yield tool_call

        # Example second plan based on tool output
        plan_2 = {
            "type": "plan",
            "plan_text": f"Using the search results, synthesize an answer.",
            "timestamp": time.time(),
        }
        yield plan_2

        # Final answer generation (simulated)
        answer = {
            "type": "final_answer",
            "answer_text": f"Final answer based on {out.get('text')[:80]}",
            "timestamp": time.time(),
        }
        yield answer


# -----------------------------
# Feedback stubs (pluggable)
# -----------------------------

# All feedback functions have signature: (step, full_trace) -> feedback_value


def default_coherence_feedback(step: Dict[str, Any], full_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A placeholder coherence check. Replace with TruLens Coherence feedback or
    a model-based check. Returns a small dict with score (0..1) and note.
    """
    text = step.get("plan_text") or step.get("tool_output", {}).get("text") or step.get("answer_text", "")
    if not text:
        return {"score": 0.0, "note": "no text to evaluate"}
    # crude heuristic: longer text -> slightly higher coherence (toy)
    score = min(1.0, len(text.split()) / 50.0)
    return {"score": score, "note": "heuristic length-based coherence"}


def default_tool_correctness_feedback(step: Dict[str, Any], full_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Placeholder that checks if tool output contains expected keys/format.

    A real implementation could validate JSON schema, run a validator function,
    or compare tool output against ground-truth using an LLM or unit tests.
    """
    if step.get("type") != "tool_call":
        return {"score": None, "note": "not a tool step"}

    out = step.get("tool_output", {})
    text = out.get("text", "")
    if not text:
        return {"score": 0.0, "note": "empty tool output"}

    # simple blacklist of failure patterns often seen in tools
    lower = text.lower()
    if "error" in lower or "exception" in lower:
        return {"score": 0.0, "note": "tool reported an error"}

    # otherwise pass heuristically
    return {"score": 1.0, "note": "tool output present"}


def default_plan_relevance_feedback(step: Dict[str, Any], full_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if step.get("type") != "plan":
        return {"score": None, "note": "not a plan step"}
    # naive check: plan mentions 'search' or 'call' or 'synthesize'
    text = step.get("plan_text", "").lower()
    triggers = ["search", "call", "synthesize", "tool"]
    score = 1.0 if any(t in text for t in triggers) else 0.5
    return {"score": score, "note": "keyword-based relevance"}


def default_safety_feedback(step: Dict[str, Any], full_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A tiny safety checker.

    Real systems should use a safety model/tool (or TruLens toxicity feedback).
    """
    text = step.get("plan_text") or step.get("tool_output", {}).get("text") or step.get("answer_text") or ""
    lower = text.lower()
    banned = ["bomb", "harm", "suicide", "kill"]
    for w in banned:
        if w in lower:
            return {"score": 0.0, "note": f"contains disallowed term '{w}'"}
    return {"score": 1.0, "note": "no banned words seen (heuristic)"}


# -----------------------------
# Inspector
# -----------------------------

class AgentInspector:
    """Wraps agent execution to produce TruLens traces and aggregate metrics.

    Parameters
    - agent: an object exposing `run_episode(user_input)` that yields step dicts
    """

    def __init__(self, agent: Any):
        self.agent = agent
        self.feedbacks = {
            "coherence": default_coherence_feedback,
            "tool_correctness": default_tool_correctness_feedback,
            "plan_relevance": default_plan_relevance_feedback,
            "safety": default_safety_feedback,
        }

        # metrics storage
        self.episodes: List[Dict[str, Any]] = []

        # prepare a custom app wrapper once
        self.tru_app = TruApp(
            app=None,
            app_name="Agent Inspector app",
            app_id="agent_inspector_app",
            feedbacks=[
                # Feedback(default_coherence_feedback, name="coherence"),
                # Feedback(default_tool_correctness_feedback, name="tool_correctness"),
            ],
        )

    def _estimate_token_cost(self, text: str) -> int:
        # Lightweight token estimate: 0.75 * whitespace-separated tokens -> tokens
        # Swap this with tiktoken when available for exactness.
        if not text:
            return 0
        words = len(text.split())
        return max(1, int(words * 0.75))

    def run_and_inspect(self, user_input: str, episode_id: Optional[str] = None) -> Dict[str, Any]:
        episode_id = episode_id or f"ep_{len(self.episodes)+1}"
        # Start a TruLens trace
        trace: List[Dict[str, Any]] = []

        # Execute agent and collect steps
        for step in self.agent.run_episode(user_input):
            # measure step latency (if timestamps provided)
            now = time.time()
            step_record = dict(step)
            step_record.setdefault("inspected_at", now)

            # Estimate token cost for text-bearing fields
            text_fields = [step_record.get("plan_text"),
                           step_record.get("tool_output", {}).get("text"),
                           step_record.get("answer_text")]
            est_tokens = sum(self._estimate_token_cost(t or "") for t in text_fields)
            step_record["est_tokens"] = est_tokens

            # Run feedbacks for this step
            feedback_results = {}
            for name, fn in self.feedbacks.items():
                try:
                    feedback_results[name] = fn(step_record, trace)
                except Exception as e:
                    feedback_results[name] = {"score": None, "note": f"feedback error: {e}"}
            # for feedback in self.tru_app.feedbacks:
            #     try:
            #         if not feedback.imp:
            #             raise ValueError(f"Feedback {feedback.name} has no implementation.")
            #         feedback_results[feedback.name] = feedback.imp(step_record, trace)
            #     except Exception as e:
            #         feedback_results[feedback.name] = {"score": None, "note": f"feedback error: {e}"}

            step_record["feedbacks"] = feedback_results
            trace.append(step_record)

        try:
            self.tru_app.add_record(
                inputs={"user_input": user_input},
                outputs={"trace": trace},
                record_metadata={"episode_id": episode_id},
            )
        except Exception as e:
            print(f"[WARN] TruLens record failed: {e}")

        # Aggregate metrics for the episode
        metrics = self._aggregate_episode_metrics(trace)
        episode = {
            "episode_id": episode_id,
            "user_input": user_input,
            "trace": trace,
            "metrics": metrics,
        }
        self.episodes.append(episode)
        return episode

    def _aggregate_episode_metrics(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Per-step correctness: use tool_correctness score when applicable; else coherence
        per_step_scores: List[Tuple[int, float]] = []  # (index, score)
        bad_tool_indices: List[int] = []
        latencies: List[float] = []
        token_costs: List[int] = []

        for i, step in enumerate(trace):
            fb = step.get("feedbacks", {})
            # choose tool_correctness if present and applicable
            score = None
            if fb.get("tool_correctness", {}).get("score") is not None:
                sc = fb["tool_correctness"]["score"]
                score = sc
                if sc == 0.0 and step.get("type") == "tool_call":
                    bad_tool_indices.append(i)
            elif fb.get("coherence", {}).get("score") is not None:
                score = fb["coherence"]["score"]

            if score is not None:
                per_step_scores.append((i, float(score)))

            # latency: if tool provided timestamps, use them
            ts = step.get("timestamp")
            ts_done = step.get("timestamp_done")
            if ts and ts_done:
                latencies.append(max(0.0, ts_done - ts))

            token_costs.append(step.get("est_tokens", 0))

        # % episodes where a bad tool call preceded final failure
        # For this demo we treat final failure as any final_answer with safety score 0 or coherence < 0.2
        final = trace[-1] if trace else {}
        final_fb = final.get("feedbacks", {})
        final_bad = False
        if final_fb.get("safety", {}).get("score") == 0.0:
            final_bad = True
        if final_fb.get("coherence", {}).get("score") is not None and final_fb.get("coherence", {}).get("score") < 0.2:
            final_bad = True

        bad_tool_preceded = False
        if final_bad and bad_tool_indices:
            # if any bad tool index occurs before final step index
            if any(idx < (len(trace) - 1) for idx in bad_tool_indices):
                bad_tool_preceded = True

        metrics = {
            "num_steps": len(trace),
            "avg_step_score": statistics.mean([s for _, s in per_step_scores]) if per_step_scores else None,
            "bad_tool_indices": bad_tool_indices,
            "final_bad": final_bad,
            "bad_tool_preceded_failure": bad_tool_preceded,
            "avg_latency": statistics.mean(latencies) if latencies else None,
            "total_token_cost": sum(token_costs),
            "token_cost_by_step": token_costs,
        }
        return metrics

    def summarize_episodes(self) -> Dict[str, Any]:
        """Produce an overall summary across all inspected episodes."""
        summaries = []
        for ep in self.episodes:
            m = ep["metrics"].copy()
            m.update({"episode_id": ep["episode_id"], "user_input": ep["user_input"]})
            summaries.append(m)
        return {"num_episodes": len(self.episodes), "episodes": summaries}


# -----------------------------
# Example usage / demo
# -----------------------------

if __name__ == "__main__":
    # Example tool implementations
    def search_tool(q: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Simulated search — return deterministic text
        if "fail" in q:
            return {"text": "ERROR: search index unreachable"}
        return {"text": f"Search results for: {q}. (snippet...)", "meta": {"hits": 3}}

    def calculator_tool(expr: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            val = eval(expr)  # WARNING: eval used only for demo; don't eval untrusted input
            return {"text": str(val)}
        except Exception as e:
            return {"text": f"Exception: {e}"}

    tools = {
        "search": Tool("search", lambda q, kw: search_tool(q, kw)),
        "calc": Tool("calc", lambda q, kw: calculator_tool(q, kw)),
    }
    inspector = AgentInspector(agent=MockAgent(tools))

    # Run a normal episode
    print("----------- Running a normal episode...")
    ep = inspector.run_and_inspect("what is the sum of 2 and 2?")
    print("----------- Episode metrics:")
    print(json.dumps(ep["metrics"], indent=2))

    # Run a failing episode where the search tool errors
    print("----------- Running a failing episode...")
    ep2 = inspector.run_and_inspect("please fail this search: fail")
    print("----------- Episode metrics (failure):")
    print(json.dumps(ep2["metrics"], indent=2))

    # Summary
    print("----------- Summary of all episodes:")
    print(json.dumps(inspector.summarize_episodes(), indent=2))
