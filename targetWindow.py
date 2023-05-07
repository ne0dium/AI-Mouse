import random
import time

import numpy as np
import pygame
import torch
import win32api
import win32con

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


class Enemy:
    def __init__(self, x, y, radius, color, speed_x, speed_y, WIDTH, HEIGHT):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.max_speed = abs(speed_x)
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.strafe = np.random.randint(0, 200)
        self.count = 0
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (self.x, self.y), self.radius)

    def move(self):
        self.count += 1
        # if self.count == self.strafe:
        #     self.count =0
        #     self.strafe = np.random.randint(0, 200)
        #     self.reverse()

        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce off the screen edges
        if self.x - self.radius <= 0 or self.x + self.radius >= self.WIDTH:
            self.speed_x = -self.speed_x
        if self.y - self.radius <= 0 or self.y + self.radius >= self.HEIGHT:
            self.speed_y = -self.speed_y

    def reverse(self):
        self.speed_x = -self.speed_x
        self.speed_y = -self.speed_y

    def reset(self):
        self.x = random.randint(self.radius, self.WIDTH - self.radius)
        self.y = random.randint(self.radius, self.HEIGHT - self.radius)
        self.speed_x = random.choice([-1, 0, 1]) * random.randint(-self.max_speed, int(self.max_speed))
        self.speed_y = random.choice([-1, 0, 1]) * random.randint(-self.max_speed, int(self.max_speed))


def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        _x = data[i:(i + seq_length)]
        _y = data[(i + seq_length):(i + seq_length) + 1]
        x.append(_x)
        y.append(_y)

    return x, y


def move(target, data, delays, predicted_delay):
    st = time.perf_counter()
    points = [[int(target[0] * targetX), int(target[1] * targetY)] for targetX, targetY, _ in data[0]]
    offsets = [[points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]] for i in range(1, len(points))]
    totalx = 0
    totaly = 0
    for pos, delay in zip(offsets, delays):
        x, y = pos
        starttime = time.perf_counter()

        while True:
            if time.perf_counter() - starttime >= delay:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
                totalx += x
                totaly += y
                break
    # time.sleep(np.random.randint(4, 6) * 0.01)
    time.sleep(0.001)
    # # hidController.moveMouse(int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    print(f"click delay diff {time.perf_counter() - st - predicted_delay:0.3f}")
    time.sleep(np.random.randint(4, 6) * 0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def run(WIDTH, HEIGHT):
    pointsList = []
    points = []
    # Colors

    # Circle settings
    circle_radius = 20
    circle_speed = 2

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Click the Circle")

    # Create an enemy
    enemy = Enemy(random.randint(circle_radius, WIDTH - circle_radius),
                  random.randint(circle_radius, HEIGHT - circle_radius), circle_radius, RED, circle_speed, circle_speed,
                  WIDTH, HEIGHT)

    # Game loop
    running = True
    lastPos = None
    last_enemy_pos = None

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Click event
            # if event.type == pygame.MOUSEBUTTONDOWN:
            #     x, y = pygame.mouse.get_pos()
            #     velocity = [x - lastPos[0], y - lastPos[1]]
            #     lastPos = [x, y]
            #     distance = ((enemy.x - x) ** 2 + (enemy.y - y) ** 2) ** 0.5
            #     # print(f"Distance between circle and cursor: {distance:.2f}")
            #     print(f"velocity{velocity}")
            #     if distance <= enemy.radius:
            #         enemy.reverse()
            if event.type == pygame.MOUSEBUTTONUP:
                pointsList.append(points)
                points = []
                lastPos = None
                enemy.reset()

        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:
            if lastPos is None:
                lastPos = pygame.mouse.get_pos()
            if last_enemy_pos is None:
                last_enemy_pos = [enemy.x, enemy.y]

            x, y = pygame.mouse.get_pos()
            velocity = [x - lastPos[0], y - lastPos[1]]

            distance = [enemy.x - x, enemy.y - y]

            enemy_velocity = [enemy.x - last_enemy_pos[0], enemy.y - last_enemy_pos[1]]
            last_enemy_pos = [enemy.x, enemy.y]

            lastPos = [x, y]
            print(f"velocity {velocity} distance {distance}")
            pygame.draw.line(screen, GREEN, (x, y), (x + velocity[0], y + velocity[1]), 4)
            points.append([distance, enemy_velocity, velocity])

        # Move and draw enemy
        enemy.move()
        enemy.draw(screen)

        pygame.display.update()
        pygame.time.Clock().tick(90)

    pygame.quit()

    return pointsList


def run_eval(WIDTH, HEIGHT, model, seq=10, model_type='lstm'):
    # Initialize Pygame
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Click the Circle")

    circle_speed_min = 4
    circle_speed = 10
    circle_radius = 5
    # Create an enemy
    enemy = Enemy(random.randint(circle_radius, WIDTH - circle_radius),
                  random.randint(circle_radius, HEIGHT - circle_radius), circle_radius, RED, circle_speed, circle_speed,
                  WIDTH, HEIGHT)

    # Game loop
    running = True
    lastPos = None
    last_enemy_pos = None

    def getInt(num):
        if num < 0:
            val = np.floor(num)
        else:
            val = np.ceil(num)
        return int(val)

    distances = []
    released = True
    startTime = time.perf_counter()
    last_loop_duration = 1

    def inside(x, y, pos):
        return pos[0] + x > 0 and pos[0] + x <= WIDTH and pos[1] + y > 0 and pos[1] + y <= HEIGHT

    while running:
        screen.fill(WHITE)
        if lastPos is None:
            lastPos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Click event
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                velocity = [x - lastPos[0], y - lastPos[1]]
                lastPos = [x, y]
                distance = ((enemy.x - x) ** 2 + (enemy.y - y) ** 2) ** 0.5
                # print(f"Distance between circle and cursor: {distance:.2f}")
                # print(f"velocity{velocity}")
                if distance <= enemy.radius:
                    # enemy.reverse()
                    print("KILLED")

            if event.type == pygame.MOUSEBUTTONUP:
                # enemy.reset()
                if event.button == 3:
                    released = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    enemy.reset()

        mouse_buttons = pygame.mouse.get_pressed()

        if lastPos is None:
            lastPos = pygame.mouse.get_pos()
        if last_enemy_pos is None:
            last_enemy_pos = [enemy.x, enemy.y]

        x, y = pygame.mouse.get_pos()
        velocity = [x - lastPos[0], y - lastPos[1]]

        enemy_velocity = [enemy.x - last_enemy_pos[0], enemy.y - last_enemy_pos[1]]

        lastPos = [x, y]

        last_enemy_pos = [enemy.x, enemy.y]

        # print(predicted_frames)

        delays = np.random.normal(0.003, 0.001, size=100)
        total_sleep = np.sum(delays) + 0.011

        # click_sleep = 0.003
        # total_sleep = click_sleep * 99 + 0.001

        predicted_frames = total_sleep // last_loop_duration - 6

        use_prediction = True
        if use_prediction:
            distance = [(enemy.x + enemy_velocity[0] * predicted_frames) - x,
                        (enemy.y + enemy_velocity[1] * predicted_frames) - y]
            # pygame.draw.line(screen, (0, 0, 0), (x, y), (x + distance[0], y + distance[1]), 4)
            pygame.draw.circle(screen, (255, 255, 0), (
                enemy.x + enemy_velocity[0] * predicted_frames, enemy.y + enemy_velocity[1] * predicted_frames), 4)
        else:
            distance = [enemy.x - x, enemy.y - y]
        distances.append(distance)

        if mouse_buttons[2] and inside(x, y, distance):
            if len(distances) > seq - 1:
                if model_type == 'lstm':
                    dist = [distances[-seq:]]
                else:
                    dist = [distances[-1]]

                dist = torch.tensor(dist, dtype=torch.float) / torch.tensor([WIDTH, HEIGHT], dtype=torch.float)

                outputs = model(dist)
                pred_vel = outputs.detach().numpy()[0]

                pygame.draw.line(screen, GREEN, (x, y), (x + velocity[0] * 10, y + velocity[1] * 10,), 4)
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, getInt(pred_vel[0] * WIDTH),
                                     getInt(pred_vel[1] * HEIGHT), 0, 0)

                # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, getInt(distance[0]*0.4 + enemy_velocity[0]), getInt(distance[1]*0.4 + enemy_velocity[1]),
                #                      0, 0)
                # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, getInt(distance[0]), getInt(distance[1]),0, 0)
                # thread = Thread(target=move2, args=(distance,data,delays,total_sleep,))
                # thread.start()

                # released = False
                pygame.draw.line(screen, GREEN, (x, y), (x + pred_vel[0] * WIDTH * 10, y + pred_vel[1] * HEIGHT * 10),
                                 4)

                # print(f"output {getInt(pred_vel[0]*WIDTH), getInt(pred_vel[1]*HEIGHT)}")

        elif mouse_buttons[0]:
            x, y = pygame.mouse.get_pos()
            velocity = [x - lastPos[0], y - lastPos[1]]
            lastPos = [x, y]
            distance = ((enemy.x - x) ** 2 + (enemy.y - y) ** 2) ** 0.5
            # print(f"Distance between circle and cursor: {distance:.2f}")
            # print(f"velocity{velocity}")
            if distance <= enemy.radius:
                # enemy.reverse()
                enemy.reset()
                # print("KILLED")

        # Move and draw enemy
        last_loop_duration = time.perf_counter() - startTime
        # print(f"{last_loop_duration:.3} ms")
        startTime = time.perf_counter()

        enemy.move()
        enemy.draw(screen)

        pygame.display.update()
        pygame.time.Clock().tick(90)

    pygame.quit()
