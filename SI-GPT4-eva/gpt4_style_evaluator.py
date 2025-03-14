from openai import OpenAI
import json
from tqdm import tqdm
import time
import os

client = OpenAI(api_key="your api")

def score_with_gpt(question: str, answer: str) -> int:
    prompt = f"""
现在我将给你一个问答对 Question 和 Answer，你需要给 Answer 评分：
1. Answer 的语言风格需要与《红楼梦》中人物对白风格一致，而不是现代中文口语风格。
2. Answer 需要完整、流畅地对 Question 作出回应，但不需要体现《红楼梦》的语义，只要能够合理回答即可。
3. 评分范围为 **0 到 10** 之间的整数，仅返回评分，不要包含任何额外内容。

请根据以下问答对进行评分：
Question: {question}
Answer: {answer}

请直接返回评分（0-10 的整数），不要包含其他内容：
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise text scoring assistant that only outputs score results."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.1
        )
        
        score_text = response.choices[0].message.content.strip()
        
        try:
            score = int(score_text)
            if 0 <= score <= 10:
                return score
            else:
                return 0
        except ValueError:
            return 0
            
    except Exception as e:
        return 0

def save_results(results: list, output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def print_statistics(scores: list):
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    min_score = min(scores)
    print("\nScoring Statistics:")
    print(f"Total samples: {len(scores)}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Highest score: {max_score}")
    print(f"Lowest score: {min_score}")

def evaluate_results(input_file: str = "result.json", output_file: str = "gpt4_scored_results.json", use_existing: bool = False):
    print(f"Reading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    if os.path.exists(output_file) and use_existing:
        print(f"Found existing score file {output_file}, continuing from last checkpoint...")
        with open(output_file, "r", encoding="utf-8") as f:
            scored_results = json.load(f)
            scored_count = len(scored_results)
            print(f"Previously scored: {scored_count}")
            results = results[scored_count:]
    else:
        if os.path.exists(output_file):
            print(f"Found existing score file {output_file}, starting fresh...")
        scored_results = []
    
    print("\nStarting evaluation...\n")
    try:
        total = len(results)
        pbar = tqdm(results, total=total, desc="Scoring progress", ncols=100, position=0, leave=True)
        for i, item in enumerate(pbar):
            question = item["question"]
            answer = item["daiyu_answer"]
            
            score = score_with_gpt(question, answer)
            item["gpt4_score"] = score
            scored_results.append(item)
            
            pbar.set_postfix({"Current Score": score, "Processed": f"{i+1}/{total}"})
            
            if (i + 1) % 5 == 0:
                save_results(scored_results, output_file)
            
            time.sleep(1)
        
        print(f"\nSaving final results to {output_file}...")
        save_results(scored_results, output_file)
        
        scores = [item["gpt4_score"] for item in scored_results]
        print_statistics(scores)
        print(f"\n✅ Evaluation complete, results saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupt detected, saving current results...")
        save_results(scored_results, output_file)
        scores = [item["gpt4_score"] for item in scored_results]
        print_statistics(scores)
        print(f"\n✅ Saved {len(scored_results)} scored results to {output_file}")
        print("You can run the program later to continue from where you left off")

if __name__ == "__main__":
    use_existing = False 
    evaluate_results(use_existing=use_existing)
