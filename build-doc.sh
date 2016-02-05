#!/bin/sh

    cd $(dirname "$0") || exit 1; for i in ./prml/*.tex; do echo "Convert $i"; ./articles/bin/mkhtml5 "$i" ;done; cd - || exit 1

