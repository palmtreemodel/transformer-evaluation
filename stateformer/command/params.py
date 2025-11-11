fields = ['static', 'inst_pos_emb', 'op_pos_emb', 'arch_emb', 'byte1', 'byte2', 'byte3', 'byte4']
field_cf = 'cover'
byte_start_pos = 4

output_langs = ['byte1', 'byte2', 'byte3', 'byte4']
trace_weights = {'byte1': 0.25, 'byte2': 0.25, 'byte3': 0.25, 'byte4': 0.25}
static_field = 'static'
inst_pos_field = 'inst_pos_emb'  # instruction positional embedding
op_pos_field = 'op_pos_emb'  # opcode/operand positional embedding
arch_field = 'arch_emb'
byte_fields = [f'byte{i}' for i in range(1, 5)]
maskable_fields = [static_field] + byte_fields
aux_fields = [inst_pos_field] + [op_pos_field] + [arch_field]
non_byte_fields = [static_field] + [inst_pos_field] + [op_pos_field] + [arch_field]