# STEP 3.6b — Before the full external test: list EVERY diagnosis code that appears in the 45,053 Chapman ECGs, with
# its count, its name from ConditionNames_SNOMED-CT.csv, and the task it is counted in. Codes that the official table
# does not name are flagged, so the label mapping can be fixed BEFORE any new result is computed.
# CPU, a few seconds; no models, no download (it reads data/chapman_meta.csv saved by Step 3.6).
import os
import pandas as pd

P3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TABLE = {270492004: "1AVB 1st degree AV block", 195042002: "2AVB 2nd degree AV block", 54016002: "2AVB1 Mobitz I",
         28189009: "2AVB2 Mobitz II", 27885002: "3AVB 3rd degree AV block", 251173003: "ABI atrial bigeminy",
         39732003: "ALS axis left shift", 284470004: "APB atrial premature beats", 164917005: "AQW abnormal Q wave",
         47665007: "ARS axis right shift", 233917008: "AVB AV block", 251199005: "CCR counterclockwise rotation",
         251198002: "CR clockwise rotation", 428417006: "ERV early repolarization", 164942001: "FQRS fQRS wave",
         698252002: "IDC/IVB intraventricular block", 426995002: "JEB junctional escape beat",
         251164006: "JPT junctional premature beat", 164909002: "LBBB / LBBBB / LFBBB (LBBB and fascicular blocks)",
         164873001: "LVH left ventricle hypertrophy", 251146004: "LVQRSAL low QRS voltage, all leads",
         251148003: "LVQRSCL low QRS voltage, chest leads", 251147008: "LVQRSLL low QRS voltage, limb leads",
         164865005: "MI / MIBW / MIFW / MILW / MISW myocardial infarction", 164947007: "PRIE PR interval extension",
         164912004: "PWC P wave change", 111975006: "QTIE QT interval extension", 446358003: "RAH right atrial hypertrophy",
         59118001: "RBBB right bundle branch block", 89792004: "RVH right ventricle hypertrophy",
         429622005: "STDD ST drop down", 164930006: "STE ST extension", 428750005: "STTC ST-T change",
         164931005: "STTU ST tilt up", 164934002: "TWC T wave change", 59931005: "TWO T wave opposite",
         164937009: "UW U wave", 11157007: "VB ventricular bigeminy", 75532003: "VEB ventricular escape beat",
         13640000: "VFW ventricular fusion wave", 17338001: "VPB ventricular premature beat",
         195060002: "VPE ventricular preexcitation", 251180001: "VET ventricular escape trigeminy",
         195101003: "WAVN / SAAWR wandering pacemaker", 74390002: "WPW", 426177001: "SB sinus bradycardia",
         426783006: "SR sinus rhythm", 164889003: "AFIB atrial fibrillation", 427084000: "ST sinus tachycardia",
         164890007: "AF atrial flutter", 427393009: "SA sinus irregularity", 426761007: "SVT supraventricular tachycardia",
         713422000: "AT atrial tachycardia", 233896004: "AVNRT", 233897008: "AVRT"}
TASKS = {"MI": {164865005}, "STTC": {428750005, 429622005, 164934002, 59931005},
         "CD": {59118001, 164909002, 270492004, 195042002, 54016002, 28189009, 27885002, 698252002, 74390002},
         "HYP": {164873001, 89792004, 446358003}}

cm = pd.read_csv(f"{P3}/data/chapman_meta.csv")
codes = cm["dx"].fillna("").apply(lambda s: [int(c) for c in str(s).split(",") if c.strip().isdigit()])
cnt = pd.Series([c for z in codes for c in z]).value_counts()
tab = pd.DataFrame({"code": cnt.index, "ECGs": cnt.values})
tab["% of ECGs"] = (100 * tab["ECGs"] / len(cm)).round(2)
tab["name in the official table"] = tab["code"].map(TABLE).fillna("*** NOT IN THE TABLE ***")
tab["counted in task"] = tab["code"].apply(lambda c: ", ".join(t for t, s in TASKS.items() if c in s) or "-")
tab.to_csv(f"{P3}/results/chapman_code_inventory.csv", index=False)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
print(f"{len(cm):,} ECGs, {len(tab)} different codes, {int((tab['name in the official table'].str.startswith('***')).sum())} "
      f"of them not named in the official table:\n")
print(tab.to_string(index=False))
