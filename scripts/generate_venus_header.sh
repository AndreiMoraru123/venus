#!/bin/bash

echo '#pragma once'
echo ''
echo '// Auto-generated main header'
echo ''

find core/ -name '*.hpp' | grep -v 'venus.hpp' | sort | while read file; do
    echo "#include <${file#core/}>"
done

echo ''