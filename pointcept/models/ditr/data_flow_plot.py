from graphviz import Digraph

def draw_full_expanded_ditr():
    dot = Digraph(comment='Full Expanded DITR Architecture', format='jpg')
    # TB: Top to Bottom (虽然是 U-Net，但 Graphviz 用 TB 配合 rank=same 更好控制层级)
    dot.attr(rankdir='TB', compound='true', dpi='300', nodesep='0.8', ranksep='1.2')
    dot.attr('node', fontname='Helvetica', style='filled,rounded', shape='box')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # ================= COLORS =================
    C_PT_BLK = '#C8E6C9'   # 绿色: Blocks
    C_PT_DATA = '#E8F5E9'  # 浅绿: Data
    C_PT_OP = '#B2DFDB'    # 蓝绿: Pooling/Unpooling
    
    C_DITR_DATA = '#FFE0B2' # 浅橙: DINO Features
    C_DITR_OP = '#FFCCBC'   # 深橙: Scatter Max
    C_INJECT = '#FFCDD2'    # 红色: Injection Linear
    C_FUSION = '#FFD700'    # 金色: Sum
    
    C_VIS = '#E1BEE7'       # 紫色: Visual Input

    # ================= INPUT & VISUAL BACKBONE =================
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(style='invis')
        c.node('IMG', 'Images', fillcolor=C_VIS)
        c.node('DINO', 'DINOv2\n(Backbone)', fillcolor=C_VIS, shape='component')
        c.node('PROJ', 'Project & Sample\n(2D->3D)', fillcolor=C_INJECT, shape='diamond')
        c.node('PC_IN', 'Input Point Cloud', fillcolor=C_PT_DATA)
        
        c.edge('IMG', 'DINO')
        c.edge('DINO', 'PROJ')
        c.edge('PC_IN', 'PROJ', label='Coords')

    # ================= LEVEL 0 (Stride 1, 32ch) =================
    with dot.subgraph(name='cluster_level0') as c:
        c.attr(rank='same')
        # Encoder 0
        c.node('DINO_L0', 'DINO Feat L0\n(N points)', fillcolor=C_DITR_DATA)
        c.node('ENC_0', 'Encoder Stage 0\n(32ch, 2 blocks)', fillcolor=C_PT_BLK)
        
        # Decoder 3 (Final)
        c.node('DEC_3_INJECT', 'Project & Sum', shape='circle', fillcolor=C_FUSION, width='1')
        c.node('DEC_3', 'Decoder Stage 3\n(Final, 64ch)', fillcolor=C_PT_BLK)
        c.node('OUTPUT', 'Segmentation Head', fillcolor='#B3E5FC')

    # ================= LEVEL 1 (Stride 2, 64ch) =================
    with dot.subgraph(name='cluster_level1') as c:
        c.attr(rank='same')
        # Ops
        c.node('SMAX_0', 'Scatter Max', fillcolor=C_DITR_OP, shape='octagon')
        c.node('GP_0', 'Grid Pool /2', fillcolor=C_PT_OP, shape='invtrapezium')
        
        # Features
        c.node('DINO_L1', 'DINO Feat L1\n(N/2 points)', fillcolor=C_DITR_DATA)
        c.node('ENC_1', 'Encoder Stage 1\n(64ch, 2 blocks)', fillcolor=C_PT_BLK)
        
        # Decoder 2
        c.node('UP_2', 'Up & Skip', fillcolor=C_PT_OP)
        c.node('DEC_2_INJECT', 'Project & Sum', shape='circle', fillcolor=C_FUSION, width='1')
        c.node('DEC_2', 'Decoder Stage 2\n(64ch)', fillcolor=C_PT_BLK)

    # ================= LEVEL 2 (Stride 4, 128ch) =================
    with dot.subgraph(name='cluster_level2') as c:
        c.attr(rank='same')
        c.node('SMAX_1', 'Scatter Max', fillcolor=C_DITR_OP, shape='octagon')
        c.node('GP_1', 'Grid Pool /4', fillcolor=C_PT_OP, shape='invtrapezium')
        
        c.node('DINO_L2', 'DINO Feat L2\n(N/8 points)', fillcolor=C_DITR_DATA)
        c.node('ENC_2', 'Encoder Stage 2\n(128ch, 2 blocks)', fillcolor=C_PT_BLK)
        
        c.node('UP_1', 'Up & Skip', fillcolor=C_PT_OP)
        c.node('DEC_1_INJECT', 'Project & Sum', shape='circle', fillcolor=C_FUSION, width='1')
        c.node('DEC_1', 'Decoder Stage 1\n(128ch)', fillcolor=C_PT_BLK)

    # ================= LEVEL 3 (Stride 8, 256ch) =================
    with dot.subgraph(name='cluster_level3') as c:
        c.attr(rank='same')
        c.node('SMAX_2', 'Scatter Max', fillcolor=C_DITR_OP, shape='octagon')
        c.node('GP_2', 'Grid Pool /8', fillcolor=C_PT_OP, shape='invtrapezium')
        
        c.node('DINO_L3', 'DINO Feat L3\n(N/32 points)', fillcolor=C_DITR_DATA)
        c.node('ENC_3', 'Encoder Stage 3\n(256ch, 6 blocks)', fillcolor=C_PT_BLK)
        
        c.node('UP_0', 'Up & Skip', fillcolor=C_PT_OP)
        c.node('DEC_0_INJECT', 'Project & Sum', shape='circle', fillcolor=C_FUSION, width='1')
        c.node('DEC_0', 'Decoder Stage 0\n(256ch)', fillcolor=C_PT_BLK)

    # ================= LEVEL 4 (Bottleneck, 512ch) =================
    with dot.subgraph(name='cluster_level4') as c:
        c.attr(rank='same')
        c.node('SMAX_3', 'Scatter Max', fillcolor=C_DITR_OP, shape='octagon')
        c.node('GP_3', 'Grid Pool /16', fillcolor=C_PT_OP, shape='invtrapezium')
        
        # 这里 DINO L4 通常不再使用，或者仅用于可视化，这里只画 Point Encoder
        c.node('ENC_4', 'Encoder Stage 4\n(Bottleneck, 512ch)', fillcolor=C_PT_BLK, width='2.5')

    # ================= CONNECTIONS =================
    
    # 1. Initialization
    dot.edge('PROJ', 'DINO_L0')
    dot.edge('PC_IN', 'ENC_0')

    # 2. Encoder Flow (Downsampling)
    # Level 0 -> 1
    dot.edge('DINO_L0', 'SMAX_0', color='orange')
    dot.edge('ENC_0', 'GP_0', color='green')
    dot.edge('GP_0', 'ENC_1', color='green')
    dot.edge('GP_0', 'SMAX_0', label='idx', style='dashed', color='green') # Index guidance
    dot.edge('SMAX_0', 'DINO_L1', color='orange')

    # Level 1 -> 2
    dot.edge('DINO_L1', 'SMAX_1', color='orange')
    dot.edge('ENC_1', 'GP_1', color='green')
    dot.edge('GP_1', 'ENC_2', color='green')
    dot.edge('GP_1', 'SMAX_1', label='idx', style='dashed', color='green')
    dot.edge('SMAX_1', 'DINO_L2', color='orange')

    # Level 2 -> 3
    dot.edge('DINO_L2', 'SMAX_2', color='orange')
    dot.edge('ENC_2', 'GP_2', color='green')
    dot.edge('GP_2', 'ENC_3', color='green')
    dot.edge('GP_2', 'SMAX_2', label='idx', style='dashed', color='green')
    dot.edge('SMAX_2', 'DINO_L3', color='orange')

    # Level 3 -> 4 (Bottleneck)
    dot.edge('ENC_3', 'GP_3', color='green')
    dot.edge('GP_3', 'ENC_4', color='green')
    # DINO usually stops or pools but isn't used in decoder from L4
    
    # 3. Decoder Flow (Upsampling & Injection)
    
    # Bottleneck -> Dec 0
    dot.edge('ENC_4', 'UP_0', color='blue') # From Deep
    dot.edge('ENC_3', 'UP_0', label='Skip', constraint='false', color='green') # Skip
    dot.edge('UP_0', 'DEC_0_INJECT', color='blue')
    dot.edge('DINO_L3', 'DEC_0_INJECT', label='Inject', color='red', penwidth='2') # DINO Inject
    dot.edge('DEC_0_INJECT', 'DEC_0', color='blue')

    # Dec 0 -> Dec 1
    dot.edge('DEC_0', 'UP_1', color='blue')
    dot.edge('ENC_2', 'UP_1', label='Skip', constraint='false', color='green')
    dot.edge('UP_1', 'DEC_1_INJECT', color='blue')
    dot.edge('DINO_L2', 'DEC_1_INJECT', label='Inject', color='red', penwidth='2')
    dot.edge('DEC_1_INJECT', 'DEC_1', color='blue')

    # Dec 1 -> Dec 2
    dot.edge('DEC_1', 'UP_2', color='blue')
    dot.edge('ENC_1', 'UP_2', label='Skip', constraint='false', color='green')
    dot.edge('UP_2', 'DEC_2_INJECT', color='blue')
    dot.edge('DINO_L1', 'DEC_2_INJECT', label='Inject', color='red', penwidth='2')
    dot.edge('DEC_2_INJECT', 'DEC_2', color='blue')

    # Dec 2 -> Dec 3 (Final)
    # 这里有点特殊，通常最后一层没有 Unpooling 模块，或者是一个 Identity Unpooling
    # 假设这里是上采样回原始分辨率
    dot.edge('DEC_2', 'DEC_3_INJECT', color='blue', label='Up')
    dot.edge('ENC_0', 'DEC_3_INJECT', label='Skip', constraint='false', color='green')
    dot.edge('DINO_L0', 'DEC_3_INJECT', label='Inject', color='red', penwidth='2')
    dot.edge('DEC_3_INJECT', 'DEC_3', color='blue')

    dot.edge('DEC_3', 'OUTPUT')

    # Save
    output_path = 'ditr_full_expanded'
    dot.render(output_path, view=False)
    print(f"Full Expanded Architecture saved to {output_path}.jpg")

if __name__ == '__main__':
    draw_full_expanded_ditr()