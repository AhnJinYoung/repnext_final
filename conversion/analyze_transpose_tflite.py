
#!/usr/bin/env python3
import argparse, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema

OPNAMES = {getattr(schema.BuiltinOperator, name): name for name in dir(schema.BuiltinOperator) if name.isupper()}
TRANSPOSE = schema.BuiltinOperator.TRANSPOSE


def tensor_data(model, tensor):
    if tensor.buffer is None or tensor.buffer < 0:
        return None
    buf = model.buffers[tensor.buffer]
    if buf.data is None or len(buf.data) == 0:
        return None
    raw = np.array(buf.data, dtype=np.uint8)
    if tensor.type == schema.TensorType.INT32:
        return raw.tobytes()
    return raw


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('tflite', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args=ap.parse_args()
    model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(args.tflite.read_bytes(),0))
    sg=model.subgraphs[0]
    opcodes=[model.operatorCodes[op.opcodeIndex].builtinCode for op in sg.operators]
    consumers=defaultdict(list)
    producer={}
    for i,op in enumerate(sg.operators):
        for t in op.inputs:
            if t is not None and t >= 0: consumers[t].append(i)
        for t in op.outputs:
            if t is not None and t >= 0: producer[t]=i
    rows=[]; patterns=Counter(); perms=Counter(); nextops=Counter(); prevops=Counter()
    identity=[]; cancel=[]
    for i,op in enumerate(sg.operators):
        if opcodes[i] != TRANSPOSE: continue
        inp=op.inputs[0]; perm_t=op.inputs[1]; out=op.outputs[0]
        perm_raw=tensor_data(model, sg.tensors[perm_t])
        perm=None
        if perm_raw is not None:
            if isinstance(perm_raw, (bytes, bytearray)):
                perm=tuple(int(x) for x in np.frombuffer(perm_raw, dtype=np.int32).tolist())
            else:
                perm=tuple(int(x) for x in perm_raw.tolist())
        perms[str(perm)] += 1
        prev=producer.get(inp)
        prev_name=OPNAMES.get(opcodes[prev], str(opcodes[prev])) if prev is not None else 'GRAPH_INPUT'
        prevops[prev_name]+=1
        next_names=[]
        for c in consumers[out]:
            next_names.append(OPNAMES.get(opcodes[c], str(opcodes[c])))
        if not next_names: next_names=['GRAPH_OUTPUT']
        for n in next_names: nextops[n]+=1
        if perm == tuple(range(len(perm))): identity.append(i)
        # consecutive inverse transpose: op -> transpose, only consumer of first output
        if len(consumers[out]) == 1:
            j=consumers[out][0]
            if opcodes[j] == TRANSPOSE:
                op2=sg.operators[j]
                p2_raw=tensor_data(model, sg.tensors[op2.inputs[1]])
                p2=None
                if isinstance(p2_raw, (bytes, bytearray)):
                    p2=np.frombuffer(p2_raw, dtype=np.int32)
                elif p2_raw is not None:
                    p2=p2_raw
                if p2 is not None:
                    p2=tuple(int(x) for x in p2.tolist())
                    if perm is not None and len(perm)==len(p2):
                        composed=tuple(perm[k] for k in p2)
                        if composed == tuple(range(len(perm))): cancel.append((i,j))
        rows.append({
            'op_index': int(i),
            'perm': list(perm) if perm is not None else None,
            'input': int(inp),
            'output': int(out),
            'prev': prev_name,
            'next': next_names,
            'out_consumers': int(len(consumers[out])),
        })
    payload={
        'file':str(args.tflite),
        'op_count':len(sg.operators),
        'transpose_count':len(rows),
        'perms':dict(perms),
        'prevops':dict(prevops),
        'nextops':dict(nextops),
        'identity':[int(x) for x in identity],
        'cancel_pairs':[[int(a), int(b)] for a, b in cancel],
        'sample':rows[:50],
    }
    text=json.dumps(payload,indent=2)
    if args.out: args.out.write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__': main()
