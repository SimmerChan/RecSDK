#!/usr/bin/env bash

export GLOO_SOCKET_IFNAME="lo"
pytest ./ --cov=hybrid_torchrec --cov-branch --cov-report=term-missing --junitxml=final.xml --cov-report=xml:coverage.xml