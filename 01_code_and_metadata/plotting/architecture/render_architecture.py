#!/usr/bin/env python3
"""Render the model architecture from its graph specification."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

HERE=Path(__file__).resolve().parent
ir=yaml.safe_load((HERE/'architecture.reviewed.yaml').read_text())
node_by_id={n['id']:n for n in ir['nodes']}
W,H=183,116
font='DejaVu Sans'
blue='#4C78A8'; pale='#EEF4FA'; orange='#D98935'; po='#FFF4E7'; gray='#647484'; ink='#243746'
plt.rcParams.update({'font.family':font,'font.size':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(W/25.4,H/25.4),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(H,0);ax.axis('off')
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
svg=ET.Element('{%s}svg'%NS,{'width':f'{W}mm','height':f'{H}mm','viewBox':f'0 0 {W} {H}','role':'img'})
def s(tag,a=None,parent=None):return ET.SubElement(parent if parent is not None else svg,'{%s}%s'%(NS,tag),{str(k):str(v) for k,v in (a or {}).items()})
s('title').text='ASPP-AG-SwinUNet, final B internal 256-pixel configuration'
s('rect',{'width':W,'height':H,'fill':'white'})
defs=s('defs')
for name,color in [('blue',blue),('orange',orange),('gray',gray)]:
    m=s('marker',{'id':name,'markerWidth':2.5,'markerHeight':2.5,'refX':2.4,'refY':1.25,'orient':'auto','markerUnits':'userSpaceOnUse'},defs)
    s('path',{'d':'M0,0 L2.5,1.25 L0,2.5 Z','fill':color},m)

texts=[]; boxes={}; drawn_edges=[];routes=[]
def txt(label,x,y,size=8,weight='normal',color=ink,anchor='middle'):
    s('text',{'x':x,'y':y,'font-family':font,'font-size':size*25.4/72,'font-weight':weight,'text-anchor':anchor,'dominant-baseline':'central','fill':color}).text=label
    ax.text(x,y,label,ha='center' if anchor=='middle' else 'left',va='center',fontsize=size,fontweight=weight,fontfamily=font,color=color)
    width=TextPath((0,0),label,prop=FontProperties(family=font,weight=weight,size=size*25.4/72)).get_extents().width
    texts.append({'text':label,'x':x,'y':y,'font_pt':size,'width_mm':float(width),'anchor':anchor})
def rect(i,x,y,w,h,fill=pale,stroke=blue):
    boxes[i]=[x,y,w,h]
    s('rect',{'id':i,'x':x,'y':y,'width':w,'height':h,'rx':1,'fill':fill,'stroke':stroke,'stroke-width':.3})
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1',facecolor=fill,edgecolor=stroke,linewidth=.85))
def circ(i,x,y,r,label):
    boxes[i]=[x-r,y-r,2*r,2*r]
    s('circle',{'id':i,'cx':x,'cy':y,'r':r,'fill':'#FFFFFF','stroke':blue,'stroke-width':.3})
    ax.add_patch(Circle((x,y),r,facecolor='white',edgecolor=blue,lw=.85))
    txt(label,x,y,7.5,weight='bold')
def route(a,b,points,kind='main',arrow=True):
    assert any(e['from']==a and e['to']==b for e in ir['edges']),(a,b)
    drawn_edges.append((a,b))
    routes.append({'from':a,'to':b,'points':points,'kind':kind})
    color=orange if kind=='conditioning' else gray if kind=='residual' else blue
    marker='orange' if kind=='conditioning' else 'gray' if kind=='residual' else 'blue'
    attrs={'d':'M '+' L '.join(f'{x},{y}' for x,y in points),'fill':'none','stroke':color,'stroke-width':.35}
    if kind=='conditioning':attrs['stroke-dasharray']='1.5 1.1'
    if arrow:attrs['marker-end']='url(#'+marker+')'
    s('path',attrs)
    for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1,ls=(0,(3,2)) if kind=='conditioning' else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if arrow else '-',mutation_scale=7,linewidth=1,color=color,linestyle=(0,(3,2)) if kind=='conditioning' else '-',shrinkA=0,shrinkB=0))

route('input','e1',[(22,14.5),(28,14.5)])
for i in range(1,4):route(f'e{i}',f'e{i+1}',[(28+40*(i-1)+27,14.5),(28+40*i,14.5)])
route('e4','aspp',[(163,22),(163,30)])
route('aspp','g3',[(163,70),(163,79),(141.5,79),(141.5,72)])
for i in (1,2,3):
    x=41.5+40*(i-1);gx=x+20
    route(f'e{i}',f'ag{i}',[(x,22),(x,32)],'residual')
    route(f'g{i}',f'cat{i}',[(gx,66),(gx,51),(x+3,51)])
    route(f'g{i}',f'ag{i}',[(gx,51),(gx,37),(x+8.5,37)],'conditioning')
    route(f'ag{i}',f'cat{i}',[(x,42),(x,48)])
    route(f'cat{i}',f'd{i}',[(x,54),(x,61)])
    if i>1:route(f'd{i}',f'g{i-1}',[(x-13.5,69),(x-17,69)])
route('d1','g0',[(41.5,77),(41.5,80.2)])
route('g0','d0',[(41.5,85.8),(41.5,89)])
route('d0','head',[(57,96.5),(74,96.5)])
route('head','resize',[(104,96.5),(119,96.5)])
route('resize','output',[(125,96.5),(143,96.5)])

rect('input',3,7,19,15,'#F8FAFC','#98A9BC');txt('RGB input',12.5,11.7,7.5,'bold');txt('256 × 256',12.5,17.4,7.2)
for i in range(1,5):
    x=28+40*(i-1);w=30 if i==4 else 27
    rect(f'e{i}',x,7,w,15)
    txt(f'Swin V2 · {i}',x+w/2,11.7,8,'bold')
    txt(node_by_id[f'e{i}']['shape'],x+w/2,17.4,7.2)
for i in (1,2,3):
    x=41.5+40*(i-1)
    rect(f'ag{i}',x-8.5,32,17,10,po,orange);txt(f'AG{i}',x,37,8,'bold')
    circ(f'cat{i}',x,51,3,'C');circ(f'g{i}',x+20,69,3,'↑2')
    rect(f'd{i}',x-13.5,61,27,16)
    txt(f'Decoder {i}',x,66.3,8,'bold');txt(node_by_id[f'd{i}']['shape'],x,72.3,7.2)
    s('circle',{'cx':x+20,'cy':51,'r':.55,'fill':blue});ax.add_patch(Circle((x+20,51),.55,color=blue))
rect('aspp',148,30,30,40,po,orange)
txt('ASPP',163,34.2,8.3,'bold')
txt('5 parallel branches',163,39,7)
txt('1 × 1',163,43.5,7.2)
txt('3 × 3: r = 1/2/3',163,48,7)
txt('Global pooling',163,52.5,7.2)
txt('Concat → 1 × 1',163,57,7.2)
s('line',{'x1':152,'x2':174,'y1':61,'y2':61,'stroke':orange,'stroke-width':.2});ax.plot([152,174],[61,61],color=orange,lw=.55)
txt(node_by_id['aspp']['shape'],163,65.5,7.2)
circ('g0',41.5,83,2.8,'↑2')
rect('d0',26,89,31,15);txt('Decoder 0',41.5,93.8,8,'bold');txt(node_by_id['d0']['shape'],41.5,99.5,7.2)
rect('head',74,89,30,15,'#F8FAFC','#98A9BC');txt('1 × 1 head',89,93.8,8,'bold');txt(node_by_id['head']['shape'],89,99.5,7.2)
circ('resize',122,96.5,3,'↑2')
rect('output',143,89,35,15,'#F8FAFC','#98A9BC');txt('Output logits',160.5,93.8,8,'bold');txt(node_by_id['output']['shape'],160.5,99.5,7.2)
txt('AG: attention gate   C: concatenate   ↑2: bilinear upsample   Tensor labels: channels × height × width',6,108,7,anchor='start',color='#5A6775')
txt('Internal model shown. Separate human model: ASPP r = 6/12/18; four output classes.',6,112.5,7,anchor='start',color='#5A6775')

assert set(drawn_edges)=={(e['from'],e['to']) for e in ir['edges']}
assert len(drawn_edges)==len(ir['edges'])
assert set(boxes)==set(node_by_id)
overflows=[]
for t in texts:
    left=t['x']-t['width_mm']/2 if t['anchor']=='middle' else t['x']
    if left<1 or left+t['width_mm']>182:overflows.append(t)
assert not overflows,overflows

out=HERE.parents[1]/'figures';out.mkdir(exist_ok=True)
ET.ElementTree(svg).write(out/'Figure2B_architecture.svg',encoding='utf-8',xml_declaration=True)
fig.savefig(out/'Figure2B_architecture.pdf',dpi=600)
fig.savefig(out/'Figure2B_architecture.png',dpi=600)
fig.savefig(out/'Figure2B_architecture-preview.png',dpi=180)
plt.close(fig)
caption='Fig. 2B. ASPP-AG-SwinUNet architecture for the final B internal segmentation model. A 256 × 256 RGB input is encoded by four Swin Transformer V2 stages with 96, 192, 384 and 768 channels at 64 × 64, 32 × 32, 16 × 16 and 8 × 8 resolutions. ASPP combines a 1 × 1 branch, three 3 × 3 branches with dilation rates 1, 2 and 3, and an image-pooling branch, followed by concatenation and 1 × 1 projection to 768 channels. Three attention gates modulate encoder skip features using upsampled deeper features before channel concatenation. Decoder stages 3, 2, 1 and 0 produce 384, 192, 96 and 64 channels, respectively. A 1 × 1 head outputs five-class logits at 128 × 128, followed by bilinear interpolation to 256 × 256. Tensor labels denote channels × height × width; dashed orange arrows denote gate guidance. The separately trained human model retains dilation rates 6, 12 and 18 and a four-class output head, including background.'
(HERE/'caption.txt').write_text(caption+'\n')
blueprint={'physical_size_mm':[W,H],'font_family':font,'minimum_font_pt':7,'nodes':boxes,'edges':routes,'text':texts,'topology_preserved':True,'ASPP_detail_view':'Five verified parallel branches are summarized inside the ASPP macro; exact internal graph is retained in architecture.reviewed.yaml.'}
(HERE/'publication_blueprint.json').write_text(json.dumps(blueprint,indent=2,ensure_ascii=False)+'\n')
paths=[HERE/'architecture.reviewed.yaml',HERE/'render_architecture.py',HERE/'caption.txt',*sorted(out.glob('Figure2B_architecture*'))]
manifest={'figure':'Figure2B_architecture','physical_size_mm':[W,H],'font':'DejaVu Sans','font_pt_min':7,'raster_dpi':600,'gpu_used':False,'editable_sources':['architecture.reviewed.yaml','render_architecture.py','../../figures/Figure2B_architecture.svg'],'files':{str(p.relative_to(HERE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'visual_review':{'status':'pending','passes':0}}
(HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'rendered','size_mm':[W,H],'drawn_nodes':len(boxes),'drawn_edges':len(drawn_edges),'topology_preserved':True}))
