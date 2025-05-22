import pickle
import os

# Path to your pickle
pkl_path = 'data/kitti/kitti_infos_val.pkl'

# Load
data = pickle.load(open(pkl_path, 'rb'))

# Depending on MMEngine version it might be a dict with a 'data_list' key
infos = data.get('data_list', data)

print(f'Loaded {len(infos)} records from {pkl_path}\n')

# Adjust these to match your setup
data_root  = 'data/kitti'
img_prefix = 'training/image_2'

for idx, rec in enumerate(infos[:10]):
    print(f'=== Record {idx} ===')
    print('Top-level keys:', list(rec.keys()))

    # 1) CAM2 → img_path
    cam2 = rec.get('images', {}).get('CAM2', {})
    fn = cam2.get('img_path', None)
    if fn:
        full = os.path.join(data_root, img_prefix, fn)
        print(f' CAM2["img_path"] = {fn!r} → exists? {os.path.exists(full)}')

    # 2) sometimes under rec["img_info"]["filename"]
    fn2 = rec.get('img_info', {}).get('filename', None)
    if fn2:
        full2 = os.path.join(data_root, fn2)
        print(f' img_info["filename"] = {fn2!r} → exists? {os.path.exists(full2)}')

    # 3) or under rec["filename"]
    fn3 = rec.get('filename', None)
    if fn3:
        full3 = os.path.join(data_root, fn3)
        print(f' filename = {fn3!r} → exists? {os.path.exists(full3)}')

    # 4) fallback—dump whole rec to see other fields
    print(' rec snippet:', {k: rec[k] for k in rec if k in ('images','img_info','filename')})
    print()
