from RLS import RLSMagCalibrator, process_file_batch
from plot import save_results

# 8 9 10 

if __name__ == '__main__':
    rls = RLSMagCalibrator()
    file = 'data.txt'
    out_html = 'calibration_plot.html'
    out_npz = 'results.npz'

    # result = process_file_batch('data.txt', rls)

    raw_arr, corr_arr, params = process_file_batch(file, rls)
    save_results(out_npz, raw_arr, corr_arr, params)
    build_plotly_html(raw_arr, corr_arr, params, out_html)

