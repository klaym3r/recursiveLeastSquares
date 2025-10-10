from RLS import RLSMagCalibrator, process_file_batch
from plot import build_plotly_html, save_results

# 8 9 10 

def test():

    x = []
    y = []
    z = []

    with open('data.txt', 'r') as f:
        data = f.readlines()
    
    for d in data:
        arr = d.split(' ')
        x.append(float(arr[8]))
        y.append(float(arr[9]))
        z.append(float(arr[10]))

    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)

    print(f"{'min':<10} max")

    print(f"{x_min:<10} {x_max}")
    print(f"{y_min:<10} {y_max}")
    print(f"{z_min:<10} {z_max}")

if __name__ == '__main__':
    rls = RLSMagCalibrator()
    file = 'data2.txt'
    out_html = 'calibration_plot.html'
    out_npz = 'results.npz'

    raw_arr, corr_arr, params = process_file_batch(file, rls)
    
    save_results(out_npz, raw_arr, corr_arr, params)
    build_plotly_html(raw_arr, corr_arr, params, out_html)

