import numpy as np
import plotly.graph_objects as go
import itertools


def read_nums(idx: int, path: str) -> list[list[float]]:
    """
    читает показания 4 магнитометров с файла
    возвращает двумерный массив чисел
    """
    with open(path, "r") as f:
        nums = []
        cnt = 0
        for line in f:
            if cnt >= 10000:
                line_arr = line.split()[2:-5:]
                tmp = []
                # 0 для акселерометров
                # 6 для магнитометров
                for i in range(idx, len(line_arr), 12):
                    for j in range(i, i + 3):
                        tmp.append(line_arr[j])
                nums.append(list(map(lambda x: float(x), tmp)))
            cnt += 1
            if cnt > 10100:
                break

    return nums


def process_coordinates(nums: list[list[float]]) -> list[list[list[float]]]:
    """
    возвращает массив points, который состоит из 4 массивов: 4 точки
    у каждой точки есть по 3 массива: x, y, z
    """
    points = [[[] for _ in range(3)] for _ in range(4)]
    for n in nums:
        for i in range(0, len(n), 3):
            idx = i // 3
            points[idx][0].append(n[i])  # x
            points[idx][1].append(n[i + 1])  # y
            points[idx][2].append(n[i + 2])  # z
    return points


def average_coords(point: list[list[float]]):
    """
    возвращает массив из 3 усреденных точек: x, y, z
    """
    return list(map(lambda x: float(np.mean(x)), point))


def find_centroid(group_of_points) -> list[float]:
    """
    находит центроид для группы 3D точек
    group_of_points: [x1,y1,z1], [x2,y2,z2], [x3,y3,z3], [x4,y4,z4]
    """
    np_points = np.array(group_of_points)
    centroid = np.mean(np_points, axis=0)
    return centroid.tolist()


def calc_distance(p1: list[float], p2: list[float]):
    """
    вычисляет евклидово расстояние между двумя 3D точками
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calc_centroids(groups):
    centroids_of_2, centroids_of_3 = [], []
    for group in groups:
        c1 = find_centroid(group)
        centroids_of_3.append(c1)
        distances = [calc_distance(p, c1) for p in group]
        farthest_point_index = np.argmax(distances)
        new_group = [group[i] for i in range(len(group)) if i != farthest_point_index]
        c2 = find_centroid(new_group)
        centroids_of_2.append(c2)

    return centroids_of_3, centroids_of_2


def calc_distance_to_centers(points, modified_centroids):
    """
    считает расстояние от точек до центров
    возвращает dict в формате {i: [dist1, dist2, ...]}
    i - номер точки
    dist1, dist2, ... - расстояние
    """
    result = {}
    for i, p in enumerate(points):
        result[i] = []
        for c in modified_centroids:
            result[i].append(float(calc_distance(p, c)))

    return result


def print_dict(d: dict):
    """
    выводит словарь в удобном формате
    """
    for k, v in d.items():
        output = f"{k}:"
        for x in v:
            output += f" {x} "
        print(output)


def find_best_group_of_size_k(
    points: list[list[float]], k: int
) -> (list[list[float]], float):
    """
    находит самую "тесную" группу заданного размера k.
    возвращает (best_group, min_avg_spread)
    """

    if k < 1 or k > len(points):
        return [], float('inf')

    all_groups = list(itertools.combinations(points, k))
    min_spread = float('inf')
    best_group = None

    for group in all_groups:
        centroid = find_centroid(group)
        spread = sum(calc_distance(p, centroid) ** 2 for p in group)

        if spread < min_spread:
            min_spread = spread
            best_group = group

    avg_spread = min_spread / (k if k > 0 else 1)
    return list(best_group) if best_group else [], avg_spread


def find_optimal_inlier_group(points: list[list[float]], threshold_factor: float = 10.0) -> (list[list[float]], int):
    """
    автоматически  находит оптимальное k (количество "хороших" точек)
    используя метод "обрыва" (Elbow Method)

    threshold_factor: во сколько раз должен увеличиться "средний разброс", чтобы мы посчитали это "обрывом"
    """

    if len(points) < 2:
        return points, len(points)

    spreads = []
    for k in range(2, len(points) + 1):
        group, avg_spread = find_best_group_of_size_k(points, k)
        spreads.append({"k": k, "group": group, "avg_spread": avg_spread})

    best_group = spreads[0]["group"]
    best_k = spreads[0]["k"]
    last_spread = spreads[0]["avg_spread"]

    for i in range(1, len(spreads)):
        current_spread = spreads[i]["avg_spread"]
        k = spreads[i]["k"]
        group = spreads[i]["group"]

        is_jump = False
        if last_spread < 1e-9:
            is_jump = current_spread > threshold_factor
        else:
            is_jump = (current_spread / last_spread) > threshold_factor

        if is_jump:
            break
        else:
            last_spread = current_spread
            best_group = group
            best_k = k

    return best_group, best_k


def build_plot(
    points: list[list[float]],
    inlier_group: list[list[float]],
    inlier_centroid: list[float],
    sensor_type: str,
    output_filename="plot.html",
):
    """
    строит график и записывает его в html
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    labels = [f"Point {i+1}" for i in range(len(points))]

    inlier_set = set(np.array(p).tobytes() for p in inlier_group)

    point_colors = []
    point_sizes = []
    point_symbols = []

    for p in points:
        if np.array(p).tobytes() in inlier_set:
            # хорошая точка
            point_colors.append("green")
            point_sizes.append(12)
            point_symbols.append("circle")
        else:
            # плохая точка
            point_colors.append("red")
            point_sizes.append(12)
            point_symbols.append("x")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            text=labels,
            mode="markers+text",
            marker=dict(
                size=point_sizes,
                color=point_colors,
                symbol=point_symbols,
                opacity=0.9,
                line=dict(width=2),
            ),
            name="Датчики (Хорошие/Выбросы)",
        )
    )

    if inlier_centroid:
        fig.add_trace(
            go.Scatter3d(
                x=[inlier_centroid[0]],
                y=[inlier_centroid[1]],
                z=[inlier_centroid[2]],
                text=["Центроид 'хороших'"],
                mode="markers+text",
                marker=dict(size=14, color="blue", symbol="diamond", opacity=1.0),
                name="Центроиды 'хороших'",
            )
        )

    title = f"Анализ выбросов: {sensor_type}"

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Ось X", yaxis_title="Ось Y", zaxis_title="Ось Z"),
        legend_title_text="Типы точек",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(output_filename)
    print(f"График успешно сохранен в файл: {output_filename}")


def main(idx: int, path: str, plot=False, output_filename=None):
    nums = read_nums(idx, path)
    points = process_coordinates(nums)
    points = list(map(lambda x: average_coords(x), points))

    sensor_type = ""
    if idx == 0:
        sensor_type = "Акселерометры"
    elif idx == 3:
        sensor_type = "Гироскоп"
    elif idx == 6:
        sensor_type = "Магнитометры"

    inlier_group, k = find_optimal_inlier_group(points, 3.0)

    inlier_centroid = []
    if inlier_group:
        inlier_centroid = find_centroid(inlier_group)

    inlier_indices = []
    for inlier_point in inlier_group:
        for i, original_point in enumerate(points):
            if np.allclose(inlier_point, original_point):
                inlier_indices.append(i + 1)
                break
    inlier_indices.sort()

    print(f"\n{sensor_type}:")
    print(f"Найдено 'хороших' точек: {k}")
    print(f"Точки: {inlier_indices}")


    if plot:
        if output_filename:
            build_plot(
                points, inlier_group, inlier_centroid, sensor_type, output_filename
            )
        else:
            build_plot(points, inlier_group, inlier_centroid, sensor_type)


if __name__ == "__main__":
    main(0, "data/data7.txt", plot=True, output_filename="axels.html")
    main(3, "data/data7.txt", plot=True, output_filename="gyro.html")
    main(6, "data/data7.txt", plot=True, output_filename="magnits.html")
