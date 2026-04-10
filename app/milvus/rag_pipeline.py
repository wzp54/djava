import os
import re
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_file(file_path):
    with open(file_path,"r",encoding="UTF-8") as f:
        return f.read()

def get_all_md_files(directory):
    EXCLUDE_DIRS = {
        ".vuepress", ".git", ".github", ".idea", ".vscode",
        "about-the-author", "books", "javaguide", "open-source-project",
        "snippets", "zhuanlan", "high-quality-technical-articles",
        "images", "media", "style", "node_modules"
    }

    # 2. 定义垃圾文件黑名单 (遇到这些文件，直接跳过)
    EXCLUDE_FILES = {
        "readme.md", "home.md", "summary.md",
        "_sidebar.md", "_navbar.md", "index.md",
        "license", "contributing.md", "todo.md"
    }
    all_files = []

    for root, dirs, files in os.walk(directory):
        dirs[:]=[d for d in dirs if d.lower() not in EXCLUDE_DIRS]
        for file in files:
            file_lower=file.lower()
            if not file_lower.endswith(".md"):
                continue
            if file_lower in EXCLUDE_FILES:
                continue
            full_path = os.path.join(root, file)
            full_path = full_path.replace("\\", "/")
            all_files.append(full_path)

    return all_files


def clean_content(content):
    if not content:
        return ""
    # --- 1. 去除 YAML Front Matter (文件头部的 --- 配置块) ---
    content = re.sub(r'^---\n.*?---\n', '', content, flags=re.DOTALL)
    content = re.sub(r'<!--\s*@include:.*?-->', '', content)
    #保留图片位置信息
    content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[图片: \1]', content)
    #替换普通链接[文本](URL) -> 文本
    content = re.sub(r'(?<!!)\[([^\]]+)\]\([^\)]+\)', r'\1', content)
    # 依然只删网页标签，保留 Java 泛型 List<String>
    content = re.sub(r'</?(div|span|p|br|a|iframe)[^>]*>', ' ', content, flags=re.IGNORECASE)
    lines = content.split('\n')
    clean_lines = []

    adhoc_spam_keywords = [
        "关注公众号", "点击领取", "扫码关注", "回复关键字",
        "后台回复", "文章首发于", "转载请注明", "Star",
        "点赞", "在看", "开源不易", "Guide哥"
    ]

    for line in lines:
        stripped_line = line.strip()

        # 4.1 关键词过滤
        is_spam = False
        for kw in adhoc_spam_keywords:
            if kw in stripped_line:
                is_spam = True
                break

        # 4.2 过滤分割线
        if re.match(r'^[-=*_]{3,}$', stripped_line):
            is_spam = True

        if not is_spam:
            clean_lines.append(line)

    content = "\n".join(clean_lines)

    # --- 5. 格式标准化 ---
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = content.replace('\u200b', '')

    return content.strip()

def split_by_h3(content,file_path):
    parts=re.split(r'(^### .+$)',content,flags=re.MULTILINE)
    parent_docs=[]
    for i in range(1,len(parts)-1,2):
        title_line=parts[i]
        body=parts[i+1]
        title=title_line.replace('###',"").strip()
        full_content=title+body
        parent_docs.append({
            "title": title,
            "content": full_content.strip(),
            "source_file": file_path
        })
    return parent_docs

def generate_parent_id(file_path,title):
    txet=file_path+title
    md5_hash=hashlib.md5(txet.encode()).hexdigest()
    return md5_hash

def process_single_file(file_path):
    content=read_file(file_path)
    clean=clean_content(content)
    parents= split_by_h3(clean,file_path)
    for parent in parents:
        parent['id']=generate_parent_id(file_path,parent['title'])
    return parents


def split_into_chunks(parent_doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " ", ""]
    )
    texts = splitter.split_text(parent_doc["content"])

    chunks = []
    for i, text in enumerate(texts):
        chunk_id = generate_chunk_id(parent_doc["id"], i)
        chunks.append({
            "id": chunk_id,
            "content": text,
            "parent_id": parent_doc["id"],
            "chunk_index": i,
            "file_path": parent_doc["source_file"]  # 加这行
        })

    return chunks
def generate_chunk_id(parent_id, chunk_index):
    """
    生成子文档的唯一ID
    """
    text=str(parent_id)+str(chunk_index)
    md5_hash=hashlib.md5(text.encode()).hexdigest()
    return md5_hash


def process_all_files(directory):
    """
    处理目录下所有 Markdown 文件
    返回：(所有父文档列表, 所有子文档列表)
    """
    all_parents = []
    all_chunks = []

    md_files = get_all_md_files(directory)

    for file_path in md_files:
        # 处理单个文件，得到父文档
        parents = process_single_file(file_path)

        # 对每个父文档生成子文档
        for parent in parents:
            chunks = split_into_chunks(parent)
            all_chunks.extend(chunks)

        all_parents.extend(parents)

    return all_parents, all_chunks





parents, chunks = process_all_files("docs")

print(f"总文件数: 221")
print(f"父文档数: {len(parents)}")
print(f"子文档数: {len(chunks)}")