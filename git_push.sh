#!/bin/bash
# add, commit with $1 or default message, and push
git add .
git commit -m "${1:-"Updated"}"
git push