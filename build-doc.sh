#!/bin/sh

    cd $(dirname $0); for i in ./prml/*; do echo "Convert $i"; ./articles/bin/mkhtml5 "$i" ;done; cd -

