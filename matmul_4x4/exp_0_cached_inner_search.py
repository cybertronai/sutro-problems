import os
import sys
import time
import json
import math

# 1. Check for STOP_SIGNAL
STOP_SIGNAL_PATH = "/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/STOP_SIGNAL_0"
if os.path.exists(STOP_SIGNAL_PATH):
    try:
        os.remove(STOP_SIGNAL_PATH)
    except Exception:
        pass
    print("STOP_SIGNAL — halting.")
    sys.exit(0)

# Import the matmul scorer
sys.path.append("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4")
import matmul

def generate_ir():
    # We want to map our variables to the optimal 1-indexed addresses
    # Variables and their sorted order:
    # 1. acc (48 reads) -> addr 1
    # 2. tmp (48 reads) -> addr 2
    # 3. sA_0 (16 reads) -> addr 3
    # 4. sA_1 (16 reads) -> addr 4
    # 5. sA_2 (16 reads) -> addr 5
    # 6. sA_3 (16 reads) -> addr 6
    # 7..22. B_k_j (4 reads each, 16 elements) -> addr 7..22
    # 23..38. C_i_j (1 read each, 16 elements) -> addr 23..38
    # 39..54. A_i_k (1 read each, 16 elements) -> addr 39..54

    acc = 1
    tmp = 2
    sA = {k: 3 + k for k in range(4)}
    
    # Map B_k_j to addresses 7 to 22
    B_at = {}
    idx = 7
    for k in range(4):
        for j in range(4):
            B_at[(k, j)] = idx
            idx += 1
            
    # Map C_i_j to addresses 23 to 38
    C_at = {}
    for i in range(4):
        for j in range(4):
            C_at[(i, j)] = idx
            idx += 1
            
    # Map A_i_k to addresses 39 to 54
    A_at = {}
    for i in range(4):
        for k in range(4):
            A_at[(i, k)] = idx
            idx += 1

    # Format the inputs and outputs
    # Inputs list: A (row-major), B (row-major)
    inputs = [A_at[(i, k)] for i in range(4) for k in range(4)] + [B_at[(k, j)] for k in range(4) for j in range(4)]
    outputs = [C_at[(i, j)] for i in range(4) for j in range(4)]
    
    lines = [",".join(map(str, inputs))]
    
    for i in range(4):
        # Cache row i of A into sA
        for k in range(4):
            lines.append(f"copy {sA[k]},{A_at[(i, k)]}")
            
        for j in range(4):
            # acc = sA_0 * B_0_j
            lines.append(f"mul {acc},{sA[0]},{B_at[(0, j)]}")
            
            # tmp = sA_1 * B_1_j; acc += tmp
            lines.append(f"mul {tmp},{sA[1]},{B_at[(1, j)]}")
            lines.append(f"add {acc},{acc},{tmp}")
            
            # tmp = sA_2 * B_2_j; acc += tmp
            lines.append(f"mul {tmp},{sA[2]},{B_at[(2, j)]}")
            lines.append(f"add {acc},{acc},{tmp}")
            
            # tmp = sA_3 * B_3_j; C_i_j = acc + tmp
            lines.append(f"mul {tmp},{sA[3]},{B_at[(3, j)]}")
            lines.append(f"add {C_at[(i, j)]},{acc},{tmp}")
            
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)

def main():
    exp_filename = "exp_0_cached_inner_search.py"
    
    # Log exp_start
    ts = time.time()
    event_start = {
        "ts": ts,
        "type": "exp_start",
        "lane": 0,
        "exp": exp_filename
    }
    with open("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/events.jsonl", "a") as f:
        f.write(json.dumps(event_start) + "\n")
        
    # Generate and score
    ir = generate_ir()
    try:
        cost = matmul.score_4x4(ir)
        print(f"Generated IR successfully! Cost: {cost}")
        
        # Check and update record
        prev_record = 800
        if cost < prev_record:
            os.makedirs("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/records", exist_ok=True)
            record_path = f"/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/records/record_{cost}_lane0.ir"
            with open(record_path, "w") as f:
                f.write(ir + "\n")
            print(f"New record saved to {record_path}")
            
            # Log new_record event
            event_record = {
                "ts": time.time(),
                "type": "new_record",
                "cost": cost,
                "prev": prev_record,
                "lane": 0,
                "file": exp_filename
            }
            with open("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/events.jsonl", "a") as f:
                f.write(json.dumps(event_record) + "\n")
    except Exception as e:
        print(f"Error during IR scoring: {e}")
        import traceback
        traceback.print_exc()
        
    # Log tokens consumed
    token_log = {
        "ts": time.time(),
        "lane": 0,
        "exp": exp_filename,
        "tokens": 1000
    }
    with open("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/sutro-problems/matmul_4x4/token_log.jsonl", "a") as f:
        f.write(json.dumps(token_log) + "\n")

if __name__ == "__main__":
    main()
