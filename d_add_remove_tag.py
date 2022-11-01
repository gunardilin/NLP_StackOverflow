import re

def add_h1_tag(str_input:str) -> str:
    """Add h1 tag to title."""
    return "<h1>{}</h1>".format(str_input)

def remove_codes(str_input:str) -> str:
    """Remove string between <pre><code> and </code></pre>"""
    str_output = re.sub("<pre.*?</code></pre>", "<p>CodE</p>", str_input, flags=re.DOTALL)
    str_output = re.sub("<code>.*?</code>", "CodE", str_output, flags=re.DOTALL)
    return str_output

def remove_links(str_input:str) -> str:
    """Remove string between <a and </a>"""
    str_output = re.sub("<a.*?</a>", "LinK", str_input, flags=re.DOTALL)
    return str_output

def remove_pre_tags(str_input:str) -> str:
    """Remove string between <pre> and </pre>
    Check Debug Log from: https://askubuntu.com/questions/1396189/windows-terminal-ssh-connect-to-host-12-3-4-56-port-22-permission-denied
    Explanation for the tag: https://www.w3schools.com/tags/tag_pre.asp
    """
    str_output = re.sub("<pre>.*?</pre>", "<p>LoG</p>", str_input, flags=re.DOTALL)
    return str_output
    
if __name__ == "__main__":
    print(add_h1_tag("This is for title."))