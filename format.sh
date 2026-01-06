python format_header.py "openjiuwen/core/memory/**/graph*/**/*.py"
isort --profile black --line-length 120 openjiuwen/core/memory/**/graph*/
black --line-length 120 openjiuwen/core/memory/**/graph*/
isort --profile black --line-length 120 format_header.py
black --line-length 120 format_header.py
git add openjiuwen/core/memory/**/graph*/
