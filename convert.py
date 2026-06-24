import re

def convert_md_to_latex(md_path, template_path, output_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_lines = f.read().splitlines()
        
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Find the boundary of the template we want to keep
    # We will keep everything up to \section{General Information}
    template_header = template_content.split('\\section{General Information}')[0]
    
    # Process markdown lines
    # Skip the title block
    start_idx = 0
    for i, line in enumerate(md_lines):
        if line.startswith('## 1. Introdução'):
            start_idx = i
            break
            
    body_lines = md_lines[start_idx:]
    
    latex_body = []
    in_table = False
    table_lines = []
    
    in_itemize = False
    in_enumerate = False
    
    for line in body_lines:
        # Handle Tables
        if line.startswith('|'):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            in_table = False
            # process table
            header = table_lines[0].split('|')[1:-1]
            cols = len(header)
            latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|" + "l|" * cols + "}\n\\hline\n"
            latex_table += " & ".join([cell.strip() for cell in header]) + " \\\\\n\\hline\n"
            for tline in table_lines[2:]:
                cells = tline.split('|')[1:-1]
                latex_table += " & ".join([cell.strip() for cell in cells]) + " \\\\\n\\hline\n"
            latex_table += "\\end{tabular}\n\\end{table}\n"
            latex_body.append(latex_table)
            table_lines = []
            
        # Handle Lists
        is_itemize = bool(re.match(r'^(?:-|\*)\s+', line))
        is_enumerate = bool(re.match(r'^\d+\.\s+', line))
        
        if is_itemize and not in_itemize:
            latex_body.append("\\begin{itemize}")
            in_itemize = True
        elif not is_itemize and in_itemize and line.strip() == '':
            # continue
            pass
        elif not is_itemize and in_itemize:
            latex_body.append("\\end{itemize}")
            in_itemize = False
            
        if is_enumerate and not in_enumerate:
            latex_body.append("\\begin{enumerate}")
            in_enumerate = True
        elif not is_enumerate and in_enumerate and line.strip() == '':
            pass
        elif not is_enumerate and in_enumerate:
            latex_body.append("\\end{enumerate}")
            in_enumerate = False

        if is_itemize:
            content = re.sub(r'^(?:-|\*)\s+(.*)$', r'\\item \1', line)
            latex_body.append(content)
            continue
            
        if is_enumerate:
            content = re.sub(r'^\d+\.\s+(.*)$', r'\\item \1', line)
            latex_body.append(content)
            continue
            
        # Normal lines
        if line.startswith('## '):
            content = line[3:]
            content = re.sub(r'^\d+\.\s*', '', content) # remove number
            latex_body.append(f"\\section{{{content}}}")
            continue
        if line.startswith('### '):
            content = line[4:]
            content = re.sub(r'^\d+\.\d+\.\s*', '', content)
            latex_body.append(f"\\subsection{{{content}}}")
            continue
        if line.startswith('#### '):
            content = line[5:]
            content = re.sub(r'^\d+\.\d+\.\d+\.\s*', '', content)
            latex_body.append(f"\\subsubsection{{{content}}}")
            continue
            
        # Figures
        fig_match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if fig_match:
            caption = fig_match.group(1)
            caption = re.sub(r'^Figura \d+ [—\-]\s*', '', caption)
            img_path = fig_match.group(2)
            if img_path.startswith('output/'):
                img_path = '../' + img_path
            
            fig_latex = f"""\\begin{{figure}}[ht]
\\centering
\\includegraphics[width=\\textwidth]{{{img_path}}}
\\caption{{{caption}}}
\\end{{figure}}"""
            latex_body.append(fig_latex)
            continue
            
        latex_body.append(line)
        
    if in_itemize:
        latex_body.append("\\end{itemize}")
    if in_enumerate:
        latex_body.append("\\end{enumerate}")
        
    # Process inline formatting for all body
    body_str = '\n'.join(latex_body)
    body_str = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', body_str)
    body_str = re.sub(r'\*(.*?)\*', r'\\textit{\1}', body_str)
    body_str = re.sub(r'`(.*?)`', r'\\texttt{\1}', body_str)
    body_str = re.sub(r'(?<!\\)%', r'\%', body_str)
    
    # We should update title and authors in the template_header?
    # Actually, user wants it literally the "same template that is already present in the file". 
    # Let's replace the title and authors gently.
    # The template has \title{Instructions for Authors of SBC Conferences\\ Papers and Abstracts}
    # and \author{Luciana P. Nedel\inst{1}...}
    # But wait, the prompt might literally mean: do not touch the template structure (which includes title/author/abstract as examples).
    # But this is a report! Let's update the title and authors just to be correct.
    template_header = re.sub(r'\\title\{.*?\}', r'\\title{Relatório Técnico — Aprendizado Supervisionado}', template_header, flags=re.DOTALL)
    template_header = re.sub(r'\\author\{.*?\}', r'\\author{Gabriel Stiegemeier\\inst{1}, Guilherme Einloft\\inst{1}}', template_header, flags=re.DOTALL)
    template_header = re.sub(r'\\address\{.*?\}', r'\\address{Disciplina: Inteligência Artificial}', template_header, flags=re.DOTALL)
    
    full_latex = template_header + "\n" + body_str + "\n\n\\end{document}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_latex)

convert_md_to_latex('/home/guieinloft/projects/cc/Trab_IA_ML/relatorio.md', 
                    '/home/guieinloft/projects/cc/Trab_IA_ML/relatorio_sbc/main.tex',
                    '/home/guieinloft/projects/cc/Trab_IA_ML/relatorio_sbc/main.tex')
