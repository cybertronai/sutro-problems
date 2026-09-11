"""Independent numeric execution of the emitted existing-v4 subset."""
import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
path=HERE.parents[1]/'submissions/1nn-v4-20260911/score_v4.py'
spec=importlib.util.spec_from_file_location('historical_semantic_machine',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


class SemanticMachine(previous.Machine):
    def step(self,instruction):
        if instruction[0].startswith('cmp_'):
            opcode,destination,a,b=instruction
            left,right=previous.bits_to_float(self.memory[a]),previous.bits_to_float(self.memory[b])
            super().step(('cmp',destination,a,b))
            self.memory[destination]=int({'eq':left==right,'ne':left!=right,'le':left<=right,
                'gt':left>right,'ge':left>=right}[opcode[4:]])
        elif instruction[0]=='div':
            _,destination,a,b=instruction
            left=previous.bits_to_float(self.memory[a])
            right=previous.bits_to_float(self.memory[b])
            # The prior machine enforces both reads-before-write and their exact
            # access charges. Division has the same operand arity as multiply.
            super().step(('mul',destination,a,b))
            self.instructions['mul']-=1
            if not self.instructions['mul']:del self.instructions['mul']
            self.instructions['div']+=1
            value=np.float32(left)/np.float32(right)
            self.memory[destination]=int(value.view(np.uint32))
        else:
            super().step(instruction)
