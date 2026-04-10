from rag_pipeline import process_all_files
from milvus_writer import bulk_index_chunks


def main():
    print("处理文件中...")
    all_parents, all_children = process_all_files("../docs")

    print(f"父文档: {len(all_parents)}, 子文档: {len(all_children)}")

    # 写入 Milvus
    print("向量化并写入 Milvus...")
    bulk_index_chunks(all_parents, all_children, batch_size=32)

    print("✅ 完成！")


if __name__ == "__main__":
    main()