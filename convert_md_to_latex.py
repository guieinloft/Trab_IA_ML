import re

def convert_md_to_latex(md_path, tex_template_path, output_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    title = ""
    authors = ""
    
    body = []
    
    in_list = False
    in_table = False
    table_aligns = []
    table_rows = []
    
    def finalize_list():
        nonlocal in_list, body
        if in_list:
            body.append("\\end{itemize}\n")
            in_list = False

    def finalize_table():
        nonlocal in_table, table_rows, table_aligns, body
        if in_table:
            body.append("\\begin{table}[ht]\n\\centering\n")
            cols = len(table_aligns) if table_aligns else (len(table_rows[0].split("&")) if table_rows else 1)
            align_str = "|" + "|".join(["l"] * cols) + "|"
            body.append(f"\\begin{{tabular}}{{{align_str}}}\n\\hline\n")
            for i, row in enumerate(table_rows):
                body.append(row + " \\\\\n")
                if i == 0:
                    body.append("\\hline\n")
            body.append("\\hline\n\\end{tabular}\n\\end{table}\n")
            in_table = False
            table_rows = []
            table_aligns = []

    for i, line in enumerate(lines):
        line = line.strip()
        
        # Extract title and authors from the beginning
        if i == 0 and line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("**Autores:**"):
            authors = line.replace("**Autores:**", "").strip()
            continue
        if line.startswith("**Disciplina:**") or line.startswith("**Dataset:**"):
            continue
            
        # Tables
        if line.startswith("|"):
            finalize_list()
            if not in_table:
                in_table = True
            
            # check if separator
            if re.match(r"^\|(\s*-+\s*\|)+$", line):
                # Count columns
                table_aligns = ["l"] * (line.count("|") - 1)
                continue
                
            row_content = [cell.strip() for cell in line.split("|")[1:-1]]
            
            # Apply inline formatting to cells
            for j in range(len(row_content)):
                # Escape & before doing other replacements
                cell = row_content[j].replace("&", "\\&")
                cell = cell.replace("%", "\\%")
                cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
                cell = re.sub(r'\*(.*?)\*', r'\\textit{\1}', cell)
                cell = re.sub(r'`(.*?)`', r'\\texttt{\1}', cell)
                row_content[j] = cell
                
            table_rows.append(" & ".join(row_content))
            continue
        else:
            finalize_table()

        # Lists
        if line.startswith("- "):
            if not in_list:
                body.append("\\begin{itemize}\n")
                in_list = True
            item_text = line[2:].strip()
            # Inline formatting
            item_text = item_text.replace("%", "\\%")
            item_text = item_text.replace("&", "\\&")
            item_text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', item_text)
            item_text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', item_text)
            item_text = re.sub(r'`(.*?)`', r'\\texttt{\1}', item_text)
            body.append(f"\\item {item_text}\n")
            continue
        else:
            finalize_list()
            
        if line == "---" or line == "":
            if not in_table and not in_list:
                body.append("\n")
            continue
            
        # Headings
        m = re.match(r"^(#{2,4})\s+(?:(?:\d+\.)+)?\s*(.*)", line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            heading_text = heading_text.replace("%", "\\%")
            heading_text = heading_text.replace("&", "\\&")
            if level == 2:
                body.append(f"\\section{{{heading_text}}}\n")
            elif level == 3:
                body.append(f"\\subsection{{{heading_text}}}\n")
            elif level == 4:
                body.append(f"\\subsubsection{{{heading_text}}}\n")
            continue
            
        # Images
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", line)
        if m:
            caption = m.group(1)
            path = m.group(2)
            # Latex escape in caption
            caption = caption.replace("%", "\\%")
            caption = caption.replace("&", "\\&")
            caption = re.sub(r'`(.*?)`', r'\\texttt{\1}', caption)
            body.append("\\begin{figure}[ht]\n\\centering\n")
            # If path points to output/ which is up a directory in relation to main.tex
            # Since main.tex is in relatorio_sbc/, the path to output/ is ../output/
            if path.startswith("output/"):
                path = "../" + path
            body.append(f"\\includegraphics[width=0.8\\textwidth]{{{path}}}\n")
            body.append(f"\\caption{{{caption}}}\n")
            body.append("\\end{figure}\n")
            continue

        # Normal text with inline formatting
        line = line.replace("%", "\\%")
        line = line.replace("&", "\\&")
        line = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
        # Avoid matching * if it's not surrounding text
        line = re.sub(r'(?<!\\)\*(.*?)(?<!\\)\*', r'\\textit{\1}', line)
        line = re.sub(r'`(.*?)`', r'\\texttt{\1}', line)

        body.append(line + "\n")
        
    finalize_list()
    finalize_table()
    
    # Read template
    with open(tex_template_path, 'r', encoding='utf-8') as f:
        template = f.read()
        
    # Replace the document contents
    doc_start = template.find("\\begin{document}")
    if doc_start == -1:
        print("Error: Could not find \\begin{document} in template.")
        return
        
    doc_end = template.find("\\end{document}")
    
    preamble = template[:doc_start]
    
    # Use lambda to avoid regex escape issues
    preamble = re.sub(r"\\title\{.*?\}", lambda m: f"\\title{{{title}}}", preamble, flags=re.DOTALL)
    
    authors_split = [a.strip() for a in authors.replace(" e ", ",").split(",")]
    author_tex = " \\and ".join([f"{a}\\inst{{1}}" for a in authors_split])
    preamble = re.sub(r"\\author\{.*?\}", lambda m: f"\\author{{{author_tex}}}", preamble, flags=re.DOTALL)
    
    addr_tex = "\\address{Instituto de Informática -- Universidade Federal do Rio Grande do Sul (UFRGS) \\\\ Porto Alegre -- RS -- Brazil}"
    preamble = re.sub(r"\\address\{.*?\}", lambda m: f"\\address{{{addr_tex}}}", preamble, flags=re.DOTALL)
    
    final_latex = preamble + "\\begin{document}\n\n\\maketitle\n\n"
    final_latex += "".join(body)
    final_latex += "\n\\end{document}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_latex)
        
    print("Conversion complete.")

convert_md_to_latex('relatorio.md', 'relatorio_sbc/main.tex', 'relatorio_sbc/main.tex')
