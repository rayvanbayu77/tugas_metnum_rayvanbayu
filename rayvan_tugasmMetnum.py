# Rayvan Bayu Abhinowo - 21120123130053
# Tugas Metnum: Sistem Persamaan Non-Linear

import numpy as np
from typing import Tuple, List
import pandas as pd

# Definisi fungsi f1 dan f2
def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

# Turunan parsial untuk Jacobian (Newton-Raphson)
def df1_dx(x, y):
    return 2*x + y

def df1_dy(x, y):
    return x

def df2_dx(x, y):
    return 3*y**2

def df2_dy(x, y):
    return 1 + 6*x*y

# Fungsi iterasi g1A dan g2A (untuk IT Jacobi dan Seidel)
def g1A(x, y):
    # Dari f1: x^2 + xy - 10 = 0 => x = sqrt(10 - xy)
    val = 10 - x*y
    if val < 0:
        return x  # Kembalikan nilai lama jika negatif
    return np.sqrt(val)

def g2A(x, y):
    # Dari f2: y + 3xy^2 - 57 = 0 => y = 57 - 3xy^2
    result = 57 - 3*x*y**2
    # Cek overflow atau nilai yang tidak masuk akal
    if abs(result) > 1e10:
        return y  # Kembalikan nilai lama jika terlalu besar
    return result

# Fungsi iterasi g1B dan g2B
def g1B(x, y):
    # Dari f1: x^2 + xy - 10 = 0 => x = (10 - x^2) / y
    if abs(y) < 1e-10:
        return x
    result = (10 - x**2) / y
    # Cek overflow
    if abs(result) > 1e10:
        return x
    return result

def g2B(x, y):
    # Dari f2: y + 3xy^2 - 57 = 0 => y = sqrt((57 - y) / (3x))
    if abs(x) < 1e-10:
        return y
    val = (57 - y) / (3*x)
    if val < 0 or val > 1e10:
        return y
    return np.sqrt(val)

# Fungsi untuk menghitung error
def error(x_new, x_old, y_new, y_old):
    return max(abs(x_new - x_old), abs(y_new - y_old))

# 1. Metode Iterasi Titik Tetap - Jacobi dengan g1A dan g2A
def jacobi_g1A_g2A(x0, y0, epsilon, max_iter=1000):
    results = []
    x, y = x0, y0
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        x_new = g1A(x, y)
        y_new = g2A(x, y)
        
        # Cek jika nilai divergen
        if abs(x_new) > 1e10 or abs(y_new) > 1e10 or np.isnan(x_new) or np.isnan(y_new):
            return results, False, i
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x, y = x_new, y_new
    
    return results, False, max_iter

# 2. Metode Iterasi Titik Tetap - Jacobi dengan g1A dan g2B
def jacobi_g1A_g2B(x0, y0, epsilon, max_iter=1000):
    results = []
    x, y = x0, y0
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        x_new = g1A(x, y)
        y_new = g2B(x, y)
        
        # Cek jika nilai divergen
        if abs(x_new) > 1e10 or abs(y_new) > 1e10 or np.isnan(x_new) or np.isnan(y_new):
            return results, False, i
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x, y = x_new, y_new
    
    return results, False, max_iter

# 3. Metode Iterasi Titik Tetap - Seidel dengan g2A dan g1B
def seidel_g2A_g1B(x0, y0, epsilon, max_iter=1000):
    results = []
    x, y = x0, y0
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        y_new = g2A(x, y)  # Update y terlebih dahulu
        x_new = g1B(x, y_new)  # Gunakan y_new untuk update x (Seidel)
        
        # Cek jika nilai divergen
        if abs(x_new) > 1e10 or abs(y_new) > 1e10 or np.isnan(x_new) or np.isnan(y_new):
            return results, False, i
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x, y = x_new, y_new
    
    return results, False, max_iter

# 4. Metode Iterasi Titik Tetap - Seidel dengan g1B dan g2B
def seidel_g1B_g2B(x0, y0, epsilon, max_iter=1000):
    results = []
    x, y = x0, y0
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        x_new = g1B(x, y)  # Update x terlebih dahulu
        y_new = g2B(x_new, y)  # Gunakan x_new untuk update y (Seidel)
        
        # Cek jika nilai divergen
        if abs(x_new) > 1e10 or abs(y_new) > 1e10 or np.isnan(x_new) or np.isnan(y_new):
            return results, False, i
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x, y = x_new, y_new
    
    return results, False, max_iter

# 5. Metode Newton-Raphson
def newton_raphson(x0, y0, epsilon, max_iter=1000):
    results = []
    x, y = x0, y0
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        # Matriks Jacobian
        J = np.array([[df1_dx(x, y), df1_dy(x, y)],
                      [df2_dx(x, y), df2_dy(x, y)]])
        
        # Vektor fungsi
        F = np.array([f1(x, y), f2(x, y)])
        
        # Cek determinan
        if abs(np.linalg.det(J)) < 1e-10:
            return results, False, i
        
        # Hitung delta
        delta = np.linalg.solve(J, -F)
        
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x, y = x_new, y_new
    
    return results, False, max_iter

# 6. Metode Secant
def secant(x0, y0, epsilon, max_iter=1000):
    results = []
    # Gunakan perturbasi kecil untuk inisialisasi
    h = 0.01
    x_prev, y_prev = x0 - h, y0 - h
    x, y = x0, y0
    
    results.append({'Iterasi': 0, 'x': x, 'y': y, 'f1': f1(x,y), 'f2': f2(x,y), 'Error': '-'})
    
    for i in range(1, max_iter + 1):
        # Aproksimasi turunan parsial
        dx = x - x_prev
        dy = y - y_prev
        
        if abs(dx) < 1e-10 or abs(dy) < 1e-10:
            return results, False, i
        
        # Aproksimasi Jacobian menggunakan secant
        df1_dx_approx = (f1(x, y) - f1(x_prev, y)) / dx
        df1_dy_approx = (f1(x, y) - f1(x, y_prev)) / dy
        df2_dx_approx = (f2(x, y) - f2(x_prev, y)) / dx
        df2_dy_approx = (f2(x, y) - f2(x, y_prev)) / dy
        
        J = np.array([[df1_dx_approx, df1_dy_approx],
                      [df2_dx_approx, df2_dy_approx]])
        
        F = np.array([f1(x, y), f2(x, y)])
        
        if abs(np.linalg.det(J)) < 1e-10:
            return results, False, i
        
        delta = np.linalg.solve(J, -F)
        
        x_new = x + delta[0]
        y_new = y + delta[1]
        
        err = error(x_new, x, y_new, y)
        results.append({'Iterasi': i, 'x': x_new, 'y': y_new, 'f1': f1(x_new, y_new), 'f2': f2(x_new, y_new), 'Error': err})
        
        if err < epsilon:
            return results, True, i
        
        x_prev, y_prev = x, y
        x, y = x_new, y_new
    
    return results, False, max_iter

# Main execution
if __name__ == "__main__":
    # Parameter awal
    x0, y0 = 1.5, 3.5
    epsilon = 0.000001
    nim_last_two = 53
    nimx = nim_last_two % 4
    
    print("="*80)
    print("SOLUSI SISTEM PERSAMAAN NON-LINEAR")
    print("="*80)
    print(f"f1(x, y) = x² + xy - 10 = 0")
    print(f"f2(x, y) = y + 3xy² - 57 = 0")
    print(f"\nNIM 2 digit terakhir: {nim_last_two}")
    print(f"NIMx = {nim_last_two} mod 4 = {nimx}")
    print(f"\nTebakan awal: x0 = {x0}, y0 = {y0}")
    print(f"Epsilon (toleransi): {epsilon}")
    print("="*80)
    
    methods = [
        ("Jacobi dengan g1A dan g2B (NIMx=1)", jacobi_g1A_g2B),
        ("Newton-Raphson", newton_raphson),
        ("Secant", secant)
    ]
    
    summary = []
    
    for method_name, method_func in methods:
        print(f"\n{'='*80}")
        print(f"METODE: {method_name}")
        print('='*80)
        
        results, converged, iterations = method_func(x0, y0, epsilon)
        
        # Tampilkan beberapa iterasi pertama dan terakhir
        df = pd.DataFrame(results)
        
        if len(results) <= 10:
            print(df.to_string(index=False))
        else:
            print("Iterasi awal:")
            print(df.head(5).to_string(index=False))
            print("\n...")
            print("\nIterasi akhir:")
            print(df.tail(5).to_string(index=False))
        
        if converged:
            final = results[-1]
            print(f"\n✓ KONVERGEN setelah {iterations} iterasi")
            print(f"Solusi: x = {final['x']:.10f}, y = {final['y']:.10f}")
            print(f"Verifikasi: f1 = {final['f1']:.2e}, f2 = {final['f2']:.2e}")
            summary.append({
                'Metode': method_name,
                'Konvergen': 'Ya',
                'Iterasi': iterations,
                'x': final['x'],
                'y': final['y'],
                'Error Akhir': final['Error']
            })
        else:
            print(f"\n✗ TIDAK KONVERGEN setelah {iterations} iterasi")
            final = results[-1]
            summary.append({
                'Metode': method_name,
                'Konvergen': 'Tidak',
                'Iterasi': iterations,
                'x': final['x'],
                'y': final['y'],
                'Error Akhir': final['Error']
            })
    
    # Ringkasan
    print(f"\n{'='*80}")
    print("RINGKASAN SEMUA METODE")
    print('='*80)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("ANALISIS KONVERGENSI DAN KECEPATAN")
    print("="*80)
    converged_methods = [s for s in summary if s['Konvergen'] == 'Ya']
    if converged_methods:
        fastest = min(converged_methods, key=lambda x: x['Iterasi'])
        print(f"Metode tercepat: {fastest['Metode']} ({fastest['Iterasi']} iterasi)")
        print(f"Solusi akhir: x = {fastest['x']:.10f}, y = {fastest['y']:.10f}")
        print(f"\nMetode untuk NIMx = {nimx}: {methods[nimx][0]}")
        nimx_result = summary[nimx]
        if nimx_result['Konvergen'] == 'Ya':
            print(f"Status: KONVERGEN dengan {nimx_result['Iterasi']} iterasi")
        else:
            print(f"Status: TIDAK KONVERGEN")
    else:
        print("Tidak ada metode yang konvergen dengan parameter yang diberikan.")