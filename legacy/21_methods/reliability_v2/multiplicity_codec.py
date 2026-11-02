"""Pure lossless representation of registered bootstrap multiplicity mappings.

Preserves count values, present-zero keys, missing keys and key order. Every
other JSON field stays in residual data. This module writes no files, invokes
no statistical kernel and grants no scientific/resource qualification.
"""
from collections.abc import Mapping
import copy
import numpy as np

FIELDS=('group_draw_counts','closure_draw_counts')
SCHEMA='M_BOOTSTRAP_MULTIPLICITIES_V1'

def _budget(value):
    if type(value) is not int or value<=0:raise ValueError('positive integer cell budget required')
    return value

def encode_counts(value,*,max_cells):
    limit=_budget(max_cells);tables=[];arrays={};used=0
    def walk(node,path):
        nonlocal used
        if isinstance(node,Mapping):
            if any(not isinstance(k,str) for k in node):raise ValueError('JSON metadata keys must be strings')
            return {k:walk(v,path+[k]) for k,v in node.items()}
        if not isinstance(node,list):return copy.deepcopy(node)
        fields=[f for f in FIELDS if any(isinstance(row,Mapping) and f in row for row in node)]
        result=[]
        for i,row in enumerate(node):
            if isinstance(row,Mapping):
                if any(not isinstance(k,str) for k in row):raise ValueError('JSON metadata keys must be strings')
                result.append({k:None if k in fields else walk(v,path+[i,k]) for k,v in row.items()})
            else:result.append(walk(row,path+[i]))
        for field in fields:
            selected=[(i,row[field]) for i,row in enumerate(node) if isinstance(row,Mapping) and field in row]
            names=set()
            for _,counts in selected:
                if not isinstance(counts,Mapping):raise ValueError('multiplicities must be a mapping')
                for key,count in counts.items():
                    if not isinstance(key,str) or not key:raise ValueError('group identity must be a nonempty string')
                    if type(count) is not int or not 0<=count<=np.iinfo(np.uint32).max:raise ValueError('invalid exact uint32 multiplicity')
                    names.add(key)
            groups=sorted(names);rows=len(selected);columns=len(groups)
            if columns>np.iinfo(np.int32).max:raise ValueError('too many group columns')
            used+=rows*columns
            if used>limit:raise ValueError('summed multiplicity cell budget exceeded')
            counts=np.zeros((rows,columns),dtype=np.uint32);order=np.full((rows,columns),-1,dtype=np.int32);index={g:i for i,g in enumerate(groups)}
            for ri,(_,mapping) in enumerate(selected):
                for position,(group,count) in enumerate(mapping.items()):
                    ci=index[group];counts[ri,ci]=count;order[ri,position]=ci
            ti=len(tables);ck=f'counts_{ti}';ok=f'order_{ti}';arrays[ck]=counts;arrays[ok]=order
            tables.append(dict(list_path=list(path),field=field,row_indices=[i for i,_ in selected],groups=groups,counts_array=ck,order_array=ok))
        return result
    residual=walk(value,[])
    return dict(schema=SCHEMA,residual=residual,tables=tables,arrays=arrays)

def _locate(value,path):
    if not isinstance(path,list):raise ValueError('invalid list path')
    node=value
    try:
        for token in path:
            if isinstance(node,list) and type(token) is int and 0<=token<len(node):node=node[token]
            elif isinstance(node,dict) and isinstance(token,str):node=node[token]
            else:raise ValueError('invalid list path')
    except (KeyError,IndexError,TypeError) as error:raise ValueError('invalid list path') from error
    if not isinstance(node,list):raise ValueError('path is not a list')
    return node

def decode_counts(bundle,*,max_cells):
    limit=_budget(max_cells)
    if not isinstance(bundle,dict) or set(bundle)!={'schema','residual','tables','arrays'} or bundle['schema']!=SCHEMA:raise ValueError('invalid multiplicity bundle')
    tables=bundle['tables'];arrays=bundle['arrays']
    if not isinstance(tables,list) or not isinstance(arrays,dict):raise ValueError('invalid tables or arrays')
    expected={name for i in range(len(tables)) for name in (f'counts_{i}',f'order_{i}')}
    if set(arrays)!=expected:raise ValueError('unexpected or missing arrays')
    result=copy.deepcopy(bundle['residual']);used=0;seen=set()
    for i,table in enumerate(tables):
        if not isinstance(table,dict) or set(table)!={'list_path','field','row_indices','groups','counts_array','order_array'}:raise ValueError('invalid table metadata')
        if table['counts_array']!=f'counts_{i}' or table['order_array']!=f'order_{i}' or table['field'] not in FIELDS:raise ValueError('invalid array/field identity')
        groups=table['groups'];indices=table['row_indices']
        if not isinstance(groups,list) or any(not isinstance(g,str) or not g for g in groups) or len(set(groups))!=len(groups):raise ValueError('duplicate or invalid groups')
        if not isinstance(indices,list) or any(type(v) is not int or v<0 for v in indices) or len(set(indices))!=len(indices):raise ValueError('invalid row indices')
        rows=len(indices);columns=len(groups);used+=rows*columns
        if used>limit:raise ValueError('summed multiplicity cell budget exceeded')
        counts=arrays[table['counts_array']];order=arrays[table['order_array']]
        if not isinstance(counts,np.ndarray) or not isinstance(order,np.ndarray) or counts.dtype!=np.uint32 or order.dtype!=np.int32 or counts.shape!=(rows,columns) or order.shape!=(rows,columns):raise ValueError('array dtype/shape mismatch')
        target=_locate(result,table['list_path']);field=table['field']
        for ri,idx in enumerate(indices):
            if idx>=len(target) or not isinstance(target[idx],dict) or field not in target[idx] or target[idx][field] is not None:raise ValueError('corrupt count placeholder')
            identity=(tuple(table['list_path']),idx,field)
            if identity in seen:raise ValueError('duplicate count placeholder')
            seen.add(identity);row=order[ri]
            if np.any(row < -1) or np.any(row >= columns):raise ValueError('invalid key order index')
            pad=np.flatnonzero(row==-1);n=int(pad[0]) if len(pad) else columns
            if np.any(row[n:]!=-1):raise ValueError('order index after padding')
            selected=row[:n]
            if len(set(map(int,selected)))!=n:raise ValueError('duplicate key order index')
            absent=np.ones(columns,dtype=bool);absent[selected]=False
            if np.any(counts[ri,absent]):raise ValueError('nonzero counts for absent keys')
            target[idx][field]={groups[int(ci)]:int(counts[ri,int(ci)]) for ci in selected}
    # Every supported count mapping in a list was replaced by None at encoding.
    # A missing/truncated table must fail closed instead of publishing nulls.
    def check_coverage(node):
        if isinstance(node,dict):
            for value in node.values():check_coverage(value)
        elif isinstance(node,list):
            for row in node:
                if isinstance(row,dict) and any(field in row and row[field] is None for field in FIELDS):
                    raise ValueError('unresolved count placeholder coverage')
                check_coverage(row)
    check_coverage(result)
    return result
