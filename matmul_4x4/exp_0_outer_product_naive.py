import os
import sys
import time
import json

# Check for STOP_SIGNAL
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
    # Memory Layout:
    # sA = 1
    # tmp = 2
    # sB = 3..6
    # C = 7..22
    # A = 23..38
    # B = 39..54
    
    A_at = lambda i, k: 23 + i * 4 + k
    B_at = lambda k, j: 39 + k * 4 + j
    C_at = lambda i, j: 7 + i * 4 + j
    sA = 1
    tmp = 2
    sB = lambda j: 3 + j
    
    inputs = [A_at(i, k) for i in range(4) for k in range(4)] + [B_at(k, j) for k in range(4) for j in range(4)]
    outputs = [C_at(i, j) for i in range(4) for j in range(4)]
    
    lines = [",".join(map(str, inputs))]
    
    for k in range(4):
        # copy B[k, j] to sB[j]
        for j in range(4):
            lines.append(f"copy {sB(j)},{B_at(k, j)}")
        
        # for each i, copy A[i, k] to sA
        for i in range(4):
            lines.append(f"copy {sA},{A_at(i, k)}")
            
            # for each j, perform multiplication, accumulating into C[i, j]
            for j in range(4):
                if k == 0:
                    lines.append(f"mul {C_at(i, j)},{sA},{sB(j)}")
                else:
                    lines.append(f"mul {tmp},{sA},{sB(j)}")
                    lines.append(f"add {C_at(i, j)},{tmp}")
                    
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)

def main():
    exp_filename = "exp_0_outer_product_naive.py"
    
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
        prev_record = 1316
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
