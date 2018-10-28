#!/bin/bash

if [[ $# != 2 ]]
then
  echo "Wrong number of parameters."
  exit 1
fi

let passed=0
let failed=0

for f in $2/*.in
do
    echo -n "File $f "

    if ./$1 <$f 2>err.txt | diff - ${f%in}out >/dev/null && diff err.txt ${f%in}err >/dev/null
    then
        echo "OK."
        passed=$(($passed+1))
    else
        echo "FAIL."
        failed=$(($failed+1))
    fi
done

echo "Tests passed: $passed"
echo "Tests failed: $failed"

rm err.txt
