import re

with open('src/fermdocs_memory/synap.py', 'r') as f:
    text = f.read()

# Let's check how search_query is being passed to the SDK.
# Currently: search_query=[query.semantic_query] if query.semantic_query else None
# If query.semantic_query is an empty string, it passes None. Does Synap SDK expect a string or list of strings?
# Wait, if `search_query` needs to be a string according to SDK docs, we might be passing a list, causing a 500 error on the server side because they expect a string.
# Let's change it to just `query.semantic_query` if it's a string.

text = text.replace('search_query=[query.semantic_query] if query.semantic_query else None,', 'search_query=query.semantic_query if query.semantic_query else None,')

with open('src/fermdocs_memory/synap.py', 'w') as f:
    f.write(text)

print("Changed search_query parameter type.")
