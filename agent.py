from query_planner import parse_search_query, parse_answer

SYSTEM_PROMPT = """You are a reasoning assistant for node classification on an Amazon product graph.
Your goal is to select the most likely category for the target node from the provided list.

Tools:
- To perform a search, use this schema exactly:
  <search> mode={local|global}, hop={1|2}, query={your query with keywords} </search>
  • mode=local: recall neighbors within 1-2 hops of the target node
  • mode=global: recall from a global nodes pool
- The graph retriever returns results inside <information>...</information>.

Reasoning protocol:
- Begin with <think>...</think> to assess if attributes are sufficient.
- Whenever you receive new information, reason inside <think>...</think>.
- Output final choice inside <answer>...</answer> only, no extra explanation."""

def run_graphsearch(anchor_id, retriever, categories, client,
                    model="deepseek-chat", max_steps=5, alpha=1.0):
    
    anchor_info = retriever.get_node_text_by_id(anchor_id)
    degree = len(retriever.neighbors.get(anchor_id, []))
    
    user_prompt = f"""Use the following information for the node classification task:
- The target product's information: {anchor_info}
- The domain knowledge: Each node represents a product connected through co-purchase relationships. Degree of target node: {degree}.
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
        answer = parse_answer(assistant_text)
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