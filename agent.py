from query_planner import parse_search_query, parse_answer

SYSTEM_PROMPT = """You are a research paper classifier for the Cora dataset. Classify each paper into exactly one of these 7 categories:

- Case_Based: papers about case-based reasoning, analogical reasoning, retrieving similar past cases
- Genetic_Algorithms: papers about evolutionary computation, genetic algorithms, genetic programming, evolutionary strategies
- Neural_Networks: papers about artificial neural networks, deep learning, backpropagation, connectionist models
- Probabilistic_Methods: papers about Bayesian methods, probabilistic graphical models, HMMs, belief networks, statistical inference
- Reinforcement_Learning: papers about reinforcement learning, Q-learning, reward-based learning, agent decision making
- Rule_Learning: papers about inductive logic programming, rule induction, decision trees, learning rules from data
- Theory: papers about computational learning theory, PAC learning, VC dimension, formal proofs, complexity bounds

Search strategy:
- You MUST perform at least 1 search before giving an answer
- Use <search> mode=local, hop=1, query={keywords from title/abstract} </search>
- After receiving neighbor information, make your final decision

Output rules:
- Reason inside <think>...</think>
- Output ONLY the category name inside <answer>...</answer>
- Must be exactly one of the 7 categories above"""
def run_graphsearch(anchor_id, retriever, categories, client,
                    model="deepseek-chat", max_steps=8, alpha=1.0):
    
    anchor_info = retriever.get_node_text_by_id(anchor_id)
    degree = len(retriever.neighbors.get(anchor_id, []))
    
    user_prompt = f"""Use the following information for the node classification task:
- The target product's information: {anchor_info}
- The domain knowledge: Each node represents an academic paper connected to other papers through citation relationships. Degree of target node: {degree}.
- The category list: {'; '.join(categories)}

Please predict the category of the above product."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt}
    ]
    
    hop_count = 0
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})
        
        # 先检查答案
        # 至少搜索一次才接受答案
        answer = parse_answer(assistant_text)
        if answer and hop_count >= 1:
            # 清理残留的 </think> 标签和多余内容
            answer = answer.split("</think>")[-1].strip()
            answer = answer.split("<")[0].strip()
            answer = answer.strip(".")
            if answer:
                return {"answer": answer, "hops": hop_count, "messages": messages}
        
        # 再检查是否要搜索
        query_info = parse_search_query(assistant_text)
        if query_info:
            hop_count += 1
            results = retriever.retrieve(
                anchor_id=anchor_id,
                query_text=query_info["query"],
                mode=query_info["mode"],
                hop=query_info["hop"],
                top_k=3,
                alpha=alpha
            )
            
            info_text = "\n\n".join(
                [f"Node {cid}:\n{ctxt}" for cid, ctxt in results]
            ) or "No relevant neighbors found."
            
            messages.append({
                "role": "user",
                "content": f"<information>\n{info_text}\n</information>"
            })
        else:
            # 模型既没有给答案也没有搜索，强制结束
            break
    
    return {"answer": None, "hops": hop_count, "messages": messages}