#!/bin/bash
# CI_TIMEOUT=50
exec "$(dirname "$0")/../../h100/active/$(basename "$0")"
