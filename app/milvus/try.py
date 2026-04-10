from faq_index import search_faq

results = search_faq('Spring Boot 自动装配原理是什么？', top_k=3)
for r in results:
    print(f'score: {r["score"]:.4f}')
    print(f'query: {r["query"]}')
    print(f'answer: {r["answer"][:100]}...')
    print('-' * 50)