import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# 1. Make sure your environment variable is set: 
# export OPENAI_API_KEY="your-api-key-here"

# Initialize gpt-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an exhaustive technical analyzer. Provide a detailed, multi-paragraph architectural evaluation breakdown for the given payload."),
    ("human", "Perform a deep-dive analysis on the scalability, potential bottlenecks, and structural resilience of this text payload:\n\n{chunk}")
])

chain = prompt | llm | StrOutputParser()

def generate_payload():
    """Generates a dense block of text to feed into the model."""
    text_block = (
        "High-throughput multi-agent orchestration frameworks demand resilient communication layers, "
        "asynchronous state synchronization via graph structures, low-latency vector retrieval mechanisms, "
        "robust fault tolerance, and automated retry loops to handle transient network and rate-limit anomalies. "
    )
    # Multiplying to create a heavy input payload
    return text_block * 2500  # Generates a substantial chunk of text (~approx 25k-30k tokens per run)

def burn_tokens_gpt4o_mini(target_tokens=1_000_000):
    print(f"🔥 Launching Token Burner mission using gpt-4o-mini. Target: {target_tokens:,} tokens...")
    
    total_tokens_burned = 0
    iteration = 1
    
    while total_tokens_burned < target_tokens:
        payload = generate_payload()
        print(payload)
        print(f"\n--- Iteration {iteration} ---")
        print(f"-> Sending payload batch to gpt-4o-mini...")
        
        try:
            response = chain.invoke({"chunk": payload})
            
            # gpt-4o-mini uses standard cl100k_base/o200k tokenizers roughly averaging ~4 chars per token.
            # We calculate actual-ish token usage based on input + output lengths.
            batch_tokens = (len(payload) + len(response)) // 4
            total_tokens_burned += batch_tokens
            
            print(f"   [+] Burned ~{batch_tokens:,} tokens. Cumulative total: ~{total_tokens_burned:,} / {target_tokens:,}")
            
            # Brief pause to respect typical RPM (Requests Per Minute) limits on free/tier-1 developer accounts
            time.sleep(1)
            iteration += 1
            
        except Exception as e:
            print(f"⚠️ Hit rate limit or error: {e}. Cooling down for 1 seconds...")
            time.sleep(1)

    print(f"\n🎯 Mission Complete! Successfully incinerated ~{total_tokens_burned:,} tokens with gpt-4o-mini.")

if __name__ == "__main__":
    burn_tokens_gpt4o_mini(1_000_000)
