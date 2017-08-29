#!/bin/bash
ssh broaddus@falcon << EOF
	python3 summarize_models.py
	exit
EOF
rsync broaddus@falcon:summary.pkl .
