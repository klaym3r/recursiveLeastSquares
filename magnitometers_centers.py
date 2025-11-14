import numpy as np
import plotly.graph_objects as go
import itertools


def read_nums(path: str) -> list[list[float]]:
    """
    читает показания 4 магнитометров с файла
    возвращает двумерный массив чисел
    """
    with open(path, "r") as f:
        nums = []
        for line in f:
            line_arr = line.split()[2:-5:]
            tmp = []
            for i in range(6, len(line_arr), 12):
                for j in range(i, i + 3):
                    tmp.append(line_arr[j])
            nums.append(list(map(lambda x: float(x), tmp)))

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


def build_plot(points: list[list[float]], centroids, modified_centroids):
    """
    строит график и записывает его в html
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    labels = [f"Point {i+1}" for i in range(len(points))]

    x_centroids = [c[0] for c in centroids]
    y_centroids = [c[1] for c in centroids]
    z_centroids = [c[2] for c in centroids]
    centroid_labels = [f"Center-3 {i+1}" for i in range(len(centroids))]

    x_mod_centroids = [c[0] for c in modified_centroids]
    y_mod_centroids = [c[1] for c in modified_centroids]
    z_mod_centroids = [c[2] for c in modified_centroids]
    mod_centroid_labels = [f"Center-2 {i+1}" for i in range(len(modified_centroids))]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            text=labels,
            mode="markers+text",
            marker=dict(size=8, color=[0, 1, 2, 3], colorscale="Viridis", opacity=0.8),
            name="Магнитометры",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_centroids,
            y=y_centroids,
            z=z_centroids,
            text=centroid_labels,
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="diamond", opacity=1.0),
            name="Центроиды (из 3)",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_mod_centroids,
            y=y_mod_centroids,
            z=z_mod_centroids,
            text=mod_centroid_labels,
            mode="markers+text",
            marker=dict(size=10, color="green", symbol="cross", opacity=1.0),
            name="Центроиды (из 2)",
        )
    )

    fig.update_layout(
        title="Координаты магнитометров и их центроиды",
        scene=dict(xaxis_title="Ось X", yaxis_title="Ось Y", zaxis_title="Ось Z"),
        legend_title_text="Типы точек",
        margin=dict(l=0, r=0, b=0, t=40),
    )

    output_filename = "magnitometer_plot.html"
    fig.write_html(output_filename)
    print(f"График успешно сохранен в файл: {output_filename}")

    fig.show()


if __name__ == "__main__":
    nums = read_nums("data/data7.txt")
    points = process_coordinates(nums)
    points = list(map(lambda x: average_coords(x), points))
    groups = list(itertools.combinations(points, 3))

    centroids_of_3, centroids_of_2 = calc_centroids(groups)

    build_plot(points, centroids_of_3, centroids_of_2)
