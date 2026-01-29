"""Script to fix test mocks by adding model_type."""
import re

# Read the file
with open('tests/test_steering_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Go through lines and add model_type after config.hidden_size assignments
output_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    output_lines.append(line)
    
    # If we see config.hidden_size and the previous lines don't have model_type
    if '.hidden_size = ' in line:
        # Check if model_type was already added in recent lines
        has_model_type = any('model_type' in output_lines[j] for j in range(max(0, len(output_lines)-5), len(output_lines)))
        
        if not has_model_type:
            # Get indentation from current line
            indent = len(line) - len(line.lstrip())
            # Insert model_type line before this one
            output_lines.insert(-1, ' ' * indent + 'mock_model.config.model_type = "llama"\n')
    
    i += 1

# Write back
with open('tests/test_steering_model.py', 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print('Fixed test mocks!')
