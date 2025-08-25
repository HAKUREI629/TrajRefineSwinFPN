import os

# List of SNR folders in order
# snr_levels = [f"/wangyunhao/cds/wangyunhao/data/BTR/mmsegform/test/512/lognorm_frame/New202502/SNR/Match/{i}dB" for i in range(-20, -30, -1)]  # ['-20dB', '-21dB', …, '-29dB']
snr_levels = [f"../outputs/seg/swin.fpnasppv5/02-28-01:27:46/testsimCBF48000/{i}dB" for i in range(20, 30, 1)]  # ['-20dB', '-21dB', …, '-29dB']
# Containers for each metric
pfa_vals = []
pd_vals  = []
ospa_vals = []

for snr in snr_levels:
    file_path = os.path.join(snr, "results.txt")
    try:
        with open(file_path, 'r') as f:
            for last_line in f:
                pass
            line = last_line.strip()
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        pfa_vals.append('')
        pd_vals.append('')
        ospa_vals.append('')
        continue

    # Parse "Key: Value" pairs separated by commas
    data = {}
    for part in line.split(','):
        if ':' in part:
            key, val = part.split(':', 1)
            data[key.strip()] = val.strip()

    # Extract metrics (assuming keys 'PFA', 'Pd', 'OSPA' exist)
    pfa_vals.append(data.get('PFA', ''))
    pd_vals.append(data.get('Pd_avg',  ''))
    ospa_vals.append(data.get('OSPA',''))

# Print tab‑separated lines for easy Excel import
print("SNR\t" + "\t".join(snr_levels))
print("PFA\t" + "\t".join(pfa_vals))
print("Pd\t"  + "\t".join(pd_vals))
print("OSPA\t"+ "\t".join(ospa_vals))
