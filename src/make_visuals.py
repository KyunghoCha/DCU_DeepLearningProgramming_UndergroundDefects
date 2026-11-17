"""Create figures from frozen evaluation records and the original images; no inference."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import NullFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["crack", "efflorescence", "spall_detach"]
KOREAN = ["균열", "백태", "박리·박락"]
COLORS = ["#e24a33", "#008fc4", "#009e73"]
RGB = [(226, 74, 51), (0, 143, 196), (0, 158, 115)]
FONT_ROOT = (Path(os.environ["CKH_ROOT"]) / "assets/fonts/nanum" if "CKH_ROOT" in os.environ
             else Path("/usr/share/fonts/truetype/nanum"))
FONT = next((path for path in (FONT_ROOT / "NanumGothic.ttf",
             Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
             ROOT / "assets/fonts/nanum/NanumGothic.ttf") if path.is_file()), None)
if FONT is None:
    raise RuntimeError("나눔고딕을 설치하거나 assets/fonts/nanum/NanumGothic.ttf에 글꼴을 두세요.")
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update({"font.family": "NanumGothic", "font.size": 13, "axes.unicode_minus": False})


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def pipeline(out, counts):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [(0.15, 1.3, 1.7, 1.35, "학습 완료 모델\n선택 가중치 고정"),
             (2.3, 2.1, 2.15, 1.25, f"검증 {counts['val']}장\n클래스별 F1 탐색"),
             (5, 2.1, 2.1, 1.25, "임계값 3개 확정\n모델·분할과 함께 저장"),
             (7.65, 1.3, 2.15, 1.35, f"테스트 {counts['test']}장\n선택값 그대로 적용")]
    for x, y, w, h, text in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fc="#f0f4f7", ec="#52708b", lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", linespacing=1.7)
    for start, end in [((1.85, 2.1), (2.3, 2.6)), ((4.45, 2.7), (5, 2.7)), ((7.1, 2.65), (7.65, 2.1))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color="#52708b"))
    ax.text(5.1, 0.75, "마스크 평가: 위치·영역 일치도       ON/OFF 평가: 결함 포함 영상 선별", ha="center")
    ax.text(5.1, 0.2, "검증 결과로 선택 → 테스트에서 최종 측정", ha="center", fontweight="bold")
    save_figure(fig, out / "pipeline.png")


def plots(out, preflight, chosen, val, test, summary, mask, grid):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        distribution = preflight["distribution"][split]
        positive = [distribution["classes"][name]["positive_images"] for name in NAMES]
        negative = [distribution["classes"][name]["negative_images"] for name in NAMES]
        ax.bar(KOREAN, positive, color=COLORS, label="해당 클래스 있음")
        ax.bar(KOREAN, negative, bottom=positive, color="#d7dee5", label="해당 클래스 없음")
        for i, (p, n) in enumerate(zip(positive, negative)):
            ax.text(i, p/2, str(p), color="white", ha="center", va="center", fontweight="bold")
            ax.text(i, p+n/2, str(n), ha="center", va="center")
        ax.set_title(f"{'검증' if split == 'val' else '테스트'} {distribution['images']}장")
        ax.set_ylabel("영상 수")
        ax.set_ylim(0, max(preflight["counts"][s] for s in ("val", "test"))+1)
    axes[1].legend(loc="upper right", fontsize=11)
    save_figure(fig, out / "dataset_composition.png")
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for cid, name in enumerate(NAMES):
        rows = [row for row in grid if row["class"] == name]
        xs, ys = [float(r["threshold"]) for r in rows], [float(r["f1"]) for r in rows]
        threshold = chosen[name]
        selected_y = ys[xs.index(threshold)]
        ax.plot(xs, ys, marker="o", color=COLORS[cid], label=f"{KOREAN[cid]} ({threshold:g})")
        ax.scatter([threshold], [selected_y], s=135, marker="*", color=COLORS[cid], edgecolor="black", zorder=5)
    ax.set(xscale="log", xlabel="신뢰도 임계값 (로그 눈금)", ylabel="검증 영상 단위 F1", ylim=(-.02, 1.05))
    ticks=sorted({float(r["threshold"]) for r in grid})
    ax.set_xticks(ticks,[f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=.25)
    ax.legend(title="클래스 (선택값)", loc="best")
    save_figure(fig, out / "val_threshold_f1.png")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for cid, (ax, name) in enumerate(zip(axes, NAMES)):
        for gt in (False, True):
            records = [row for row in val["images"] if (cid in row["gt_classes"]) == gt]
            xs = [row["max_confidence"][cid] if row["max_confidence"][cid] is not None else .00004 for row in records]
            jitter = np.linspace(-.10, .10, len(xs))
            ax.scatter(xs, np.full(len(xs), int(gt))+jitter, c=COLORS[cid] if gt else "#7c8792",
                       marker="o" if gt else "x", s=45, alpha=.9)
        ax.axvline(chosen[name], color=COLORS[cid], linestyle="--", label=f"선택 {chosen[name]:g}")
        ax.set(yticks=[0, 1], yticklabels=["정답 OFF", "정답 ON"], ylim=(-.4, 1.4), title=KOREAN[cid])
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(axis="x", alpha=.2)
    axes[-1].set(xscale="log", xlim=(.000025, 1.1), xlabel="영상·클래스별 최대 confidence (없음은 맨 왼쪽)")
    axes[-1].set_xticks([.00004, .0001, .001, .01, .1, 1], ["없음", ".0001", ".001", ".01", ".1", "1"])
    for ax in axes: ax.xaxis.set_minor_formatter(NullFormatter())
    save_figure(fig, out / "val_confidence.png")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    positions = np.arange(4)
    values = [mask["mask"]["per_class"][name] for name in NAMES] + [mask["mask"]["overall"]]
    for offset, key, label, color in [(-.19, "map50", "mask AP50", "#3579a7"),
                                      (.19, "map50_95", "mask AP50-95", "#7fb0bd")]:
        bars = ax.bar(positions+offset, [r[key] for r in values], .36, label=label, color=color)
        ax.bar_label(bars, labels=[f"{r[key]:.3f}" for r in values], padding=3, fontsize=12)
    ax.set(xticks=positions, xticklabels=KOREAN+["전체 평균"], ylim=(0, 1.1), ylabel="마스크 AP / mAP")
    ax.legend()
    save_figure(fig, out / "test_mask_metrics.png")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for cid, (ax, row) in enumerate(zip(axes, summary["classes"])):
        values = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        ax.imshow(values, vmin=0, vmax=len(test["images"]), cmap="Blues")
        for y in range(2):
            for x in range(2):
                ax.text(x, y, f"{[['TN','FP'],['FN','TP']][y][x]}\n{values[y,x]}", ha="center", va="center", fontsize=13)
        ax.set(xticks=[0,1], yticks=[0,1], xticklabels=["OFF", "ON"], yticklabels=["OFF", "ON"],
               xlabel="예측", ylabel="정답", title=f"{KOREAN[cid]} · F1 {row['f1']:.3f}")
    save_figure(fig, out / "test_onoff_confusion.png")


def thin_mask(out):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    records = []
    for ax, width in zip(axes, (4, 20)):
        gt = np.zeros((40, 40), bool)
        pred = gt.copy()
        left = (40-width)//2
        gt[4:36, left:left+width] = True
        pred[4:36, left+2:left+width+2] = True
        canvas = np.ones((40,40,3))
        canvas[gt] = [.25,.61,.82]
        canvas[pred] = [.96,.62,.2]
        canvas[gt & pred] = [.33,.48,.4]
        intersection, union = int((gt & pred).sum()), int((gt | pred).sum())
        iou = intersection / union
        records.append({"width_px": width, "shift_px": 2, "intersection": intersection, "union": union, "iou": iou})
        ax.imshow(canvas, interpolation="nearest")
        ax.set_title(f"폭 {width}px · 옆으로 2px 이동\nIoU = {intersection}/{union} = {iou:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("픽셀 마스크의 합성 예시: 같은 이동량에서 폭에 따른 IoU 차이", fontsize=12)
    fig.subplots_adjust(top=.78)
    save_figure(fig, out / "thin_mask_iou.png")
    return records


def gt_instances(path):
    result = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = [float(x) for x in line.split()]
        result.append({"class_id": int(row[0]), "polygon_xyn": list(zip(row[1::2], row[2::2]))})
    return result


def contour_components(points):
    """Recover closed contours from masks.xyn's forward/backward connecting edges."""
    points=[tuple(point) for point in points]
    if not points: return []
    stack,positions,components=[],{},[]
    for point in points+[points[0]]:
        if point in positions:
            start=positions[point]
            contour=stack[start:]
            if len(contour)>=3:
                area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(contour,contour[1:]+contour[:1]))
                if abs(area)>1e-12: components.append(contour)
            for old in stack[start+1:]: del positions[old]
            stack=stack[:start+1]
        else:
            positions[point]=len(stack)
            stack.append(point)
    return components


def overlay(original, instances):
    layer = Image.new("RGBA", original.size)
    draw = ImageDraw.Draw(layer)
    width, height = original.size
    for item in instances:
        color = RGB[item["class_id"]]
        original_polygon=[(max(0,min(width-1,x*width)),max(0,min(height-1,y*height))) for x,y in item["polygon_xyn"]]
        if len(original_polygon)>=3:
            draw.polygon(original_polygon,fill=(*color,55))
        for contour in contour_components(item["polygon_xyn"]):
            polygon = [(max(0,min(width-1,x*width)), max(0,min(height-1,y*height))) for x,y in contour]
            draw.line(polygon+[polygon[0]], fill=(*color, 255), width=max(2, round(width/700)), joint="curve")
    return Image.alpha_composite(original.convert("RGBA"), layer).convert("RGB")


def triptych(original, gt, pred, path, crop=None, focus=None):
    panels = [original, overlay(original, gt), overlay(original, pred)]
    if crop is not None:
        panels = [panel.crop(crop) for panel in panels]
    panel_width = 800
    panel_height = round(panels[0].height / panels[0].width * panel_width)
    # Portrait views remain proportional; the report controls their final physical size.
    total = Image.new("RGB", (panel_width*3+40, panel_height+132), "white")
    draw = ImageDraw.Draw(total)
    font = ImageFont.truetype(str(FONT), 42)
    labels=["원본", "정답 폴리곤", "선택 임계값 예측"] if focus is None else ["원본",f"{KOREAN[focus]} 정답",f"{KOREAN[focus]} 예측"]
    for i, (panel, label) in enumerate(zip(panels,labels)):
        x = i*(panel_width+20)
        draw.text((x+panel_width/2, 29), label, anchor="mm", fill="#20252b", font=font)
        total.paste(panel.resize((panel_width,panel_height), Image.Resampling.LANCZOS), (x,65))
    for i, label in enumerate(KOREAN):
        x = 35 + i*800
        draw.rectangle((x,panel_height+87,x+33,panel_height+120),fill=RGB[i])
        draw.text((x+48,panel_height+82),label,fill="#20252b",font=font)
    path.parent.mkdir(parents=True,exist_ok=True)
    total.save(path,"JPEG",quality=92,subsampling=0,optimize=True)


def crop_box(instances, size):
    width,height = size
    points = [(x*width,y*height) for item in instances for x,y in item["polygon_xyn"]]
    if not points:
        return (0,0,width,height)
    xs,ys = zip(*points)
    margin = max(45, round(min(width,height)*.06))
    left,right = max(0,min(xs)-margin),min(width,max(xs)+margin)
    top,bottom = max(0,min(ys)-margin),min(height,max(ys)+margin)
    # Give very narrow defects enough neighboring surface to judge their appearance.
    if right-left < (bottom-top)*.9:
        center=(left+right)/2; half=(bottom-top)*.45
        left,right=max(0,center-half),min(width,center+half)
    return tuple(map(round,(left,top,right,bottom)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",type=Path,default=ROOT/"results")
    parser.add_argument("--dataset",type=Path,default=ROOT/"work/dataset")
    parser.add_argument("--manifest",type=Path,default=ROOT/"data/split_manifest.csv")
    parser.add_argument("--output",type=Path,default=ROOT/"docs/images")
    args=parser.parse_args()
    evaluation,dataset,out=args.evaluation.resolve(),args.dataset.resolve(),args.output.resolve()
    if not out.is_relative_to(ROOT): raise ValueError("Visual output must be inside the project")
    for stage in ("tune","test"):
        if load(evaluation/stage/"completed.json")["status"]!="completed": raise ValueError("Evaluation is incomplete")
    data_summary=load(ROOT/"data/summary.json")
    preflight={"counts":{split:row["images"] for split,row in data_summary["splits"].items()},
        "distribution":{split:{"images":row["images"],"classes":{name:{"positive_images":row["class_images"][name],
        "negative_images":row["images"]-row["class_images"][name]} for name in NAMES}}
        for split,row in data_summary["splits"].items() if split in ("val","test")}}
    selected=load(evaluation/"tune/selected_thresholds.json")
    val=load(evaluation/"tune/validation_predictions.json")
    test=load(evaluation/"test/test_predictions.json")
    summary=load(evaluation/"test/test_summary.json")
    mask=load(evaluation/"test/segmentation_metrics.json")
    with (evaluation/"tune/validation_thresholds.csv").open() as stream: grid=list(csv.DictReader(stream))
    if not (val["freeze"]==test["freeze"]==summary["freeze"]==selected["freeze"]): raise ValueError("Evaluation baselines differ")
    if sha(evaluation/"tune/validation_predictions.json")!=selected["validation_cache_sha256"]: raise ValueError("Validation cache changed")
    for split,cache in (("val",val),("test",test)):
        if len(cache["images"])!=preflight["counts"][split]: raise ValueError("Cached image count differs")
        for cid,name in enumerate(NAMES):
            if sum(cid in x["gt_classes"] for x in cache["images"])!=preflight["distribution"][split]["classes"][name]["positive_images"]:
                raise ValueError("Cached GT class distribution differs")
    with args.manifest.open(encoding="utf-8-sig",newline="") as stream:
        source_rows={row["output_image"]:row for row in csv.DictReader(stream)}
    out.mkdir(parents=True,exist_ok=True)
    pipeline(out,preflight["counts"])
    plots(out,preflight,selected["thresholds"],val,test,summary,mask,grid)
    synthetic=thin_mask(out)
    cases=[]
    counters={name:{"FP":0,"FN":0,"TP":0} for name in NAMES}
    for index,row in enumerate(test["images"],1):
        ident=f"T{index:02d}"
        image_path=dataset/row["image"]
        label_path=dataset/row["label"]
        if sha(image_path)!=row["image_sha256"] or sha(label_path)!=row["label_sha256"]: raise ValueError("Source file changed")
        with Image.open(image_path) as source: original=source.convert("RGB")
        if list(original.size[::-1])!=row["shape_hw"]: raise ValueError("Prediction image shape differs")
        gt=gt_instances(label_path)
        pred=[x for x in row["instances"] if x["confidence"]>=selected["thresholds"][NAMES[x["class_id"]]]]
        path=out/f"case_{ident}.jpg"
        triptych(original,gt,pred,path)
        decisions={}
        crops=[]
        for cid,name in enumerate(NAMES):
            positive=cid in row["gt_classes"]
            conf=row["max_confidence"][cid]
            on=conf is not None and conf>=selected["thresholds"][name]
            decision=("TP" if positive else "FP") if on else ("FN" if positive else "TN")
            decisions[name]={"decision":decision,"confidence":conf,"threshold":selected["thresholds"][name]}
            if decision in counters[name] and counters[name][decision] < (1 if decision=="TP" else 3):
                counters[name][decision]+=1
                focused_gt=[x for x in gt if x["class_id"]==cid]
                focused_pred=[x for x in pred if x["class_id"]==cid]
                box=crop_box(focused_gt+focused_pred,original.size)
                folder="tp" if decision=="TP" else "errors"
                crop_path=out/folder/f"{name}_{decision}_{counters[name][decision]}.jpg"
                triptych(original,focused_gt,focused_pred,crop_path,box,focus=cid)
                crops.append({"path":crop_path.relative_to(ROOT).as_posix(),"class":name,"decision":decision,
                              "crop_xyxy_pixels":box})
        cases.append({"id":ident,"path":path.relative_to(ROOT).as_posix(),"source_image":source_rows[row["image"]]["source_image"],
                      "derived_image":row["image"],"source_sha256":row["image_sha256"],"decisions":decisions,"crops":crops})
    guide_root=out/"guideline"
    guide_root.mkdir(exist_ok=True)
    for name in ("C1","E1","S1","N1","N4","N3"):
        source=ROOT/f"assets/guideline/{name}.jpg"
        if source.is_file():
            with Image.open(source) as im:
                im=im.convert("RGB")
                im.thumbnail((550,550),Image.Resampling.LANCZOS)
                im.save(guide_root/f"{name}.jpg",quality=90,subsampling=0,optimize=True)
        elif not (guide_root/f"{name}.jpg").is_file():
            raise FileNotFoundError(f"Missing packaged guideline example: {name}")
    inputs={p.relative_to(ROOT).as_posix():sha(p) for p in evaluation.rglob("*.json") if p.is_file()}
    outputs=[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)}
             for p in sorted(out.rglob("*")) if p.suffix in {".png",".jpg"}]
    manifest={"inputs":inputs,"freeze":selected["freeze"],"selected_thresholds":selected["thresholds"],"comparison_columns":["original","ground_truth","prediction"],
              "polygon_display":"Closed contour components; zero-area round-trip connecting edges are not drawn as boundaries",
              "prediction_filter":"confidence >= selected threshold for that class", "synthetic_iou_examples":synthetic,
              "cases":cases,"outputs":outputs}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"figures":len(outputs),"cases":len(cases),"error_crops":counters},ensure_ascii=False))


if __name__=="__main__": main()
