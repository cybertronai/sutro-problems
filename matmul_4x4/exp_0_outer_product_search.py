import os
import sys
import time
import json
import itertools

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

def generate_ir(j_copy_order, i_order, j_mul_orders):
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
        # copy B[k, j] to sB[j] according to j_copy_order
        for j in j_copy_order:
            lines.append(f"copy {sB(j)},{B_at(k, j)}")
        
        # for each i according to i_order, copy A[i, k] to sA
        for i in i_order:
            lines.append(f"copy {sA},{A_at(i, k)}")
            
            # for each j according to j_mul_orders[k][i] or similar
            # For simplicity, let's use a fixed j_mul_order for this k, i
            j_mul_order = j_mul_orders[k][i]
            for j in j_mul_order:
                if k == 0:
                    lines.append(f"mul {C_at(i, j)},{sA},{sB(j)}")
                else:
                    lines.append(f"mul {tmp},{sA},{sB(j)}")
                    lines.append(f"add {C_at(i, j)},{tmp}")
                    
    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)

def main():
    exp_filename = "exp_0_outer_product_search.py"
    
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
        
    best_cost = 999999
    best_config = None
    
    # Let's try some simple permutations to see if cost changes
    # Permutations of [0, 1, 2, 3]
    perms = list(itertools.permutations([0, 1, 2, 3]))
    
    # We will test a few configs to see if cost is invariant
    print("Testing permutations...")
    tested = 0
    for j_copy in perms[:3]: # try a few
        for i_ord in perms[:3]:
            # For j_mul_orders, we can just use a fixed permutation for all k, i
            for j_mul in perms[:3]:
                j_mul_orders = {k: {i: j_mul for i in range(4)} for k in range(4)}
                ir = generate_ir(j_copy, i_ord, j_mul_orders)
                try:
                    cost = matmul.score_4x4(ir)
                    tested += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_config = (j_copy, i_ord, j_mul)
                except Exception as e:
                    print(f"Error: {e}")
                    
    print(f"Tested {tested} permutations. Best cost found: {best_cost} with config: {best_config}")
    
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
