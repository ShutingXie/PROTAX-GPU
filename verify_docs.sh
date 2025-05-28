#!/bin/bash

echo "Verifying PROTAX-GPU documentation..."

# 1. Check if required files exist
echo "Checking file structure..."
required_files=(
    "source/conf.py"
    "source/index.rst" 
    "source/installation.rst"
    "source/usage.rst"
    "source/examples.rst"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file missing"
    fi
done

# 2. Check RST syntax
echo "Checking RST syntax..."
python -c "
import docutils.core
import sys
try:
    with open('source/index.rst', 'r') as f:
        docutils.core.publish_doctree(f.read())
    print('✅ RST syntax valid')
except Exception as e:
    print(f'❌ RST syntax error: {e}')
    sys.exit(1)
"

# 3. Try building documentation
echo "Building documentation..."
if sphinx-build -b html source build/html -q; then
    echo "✅ Documentation built successfully"
    echo "📂 View at: file://$(pwd)/build/html/index.html"
else
    echo "❌ Documentation build failed"
    exit 1
fi

# 4. Check for broken links (optional)
echo "Checking internal links..."
sphinx-build -b linkcheck source build/linkcheck -q

echo "Documentation verification complete!"