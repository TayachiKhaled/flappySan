import pygame
import random
import numpy as np
from NeuralNetwork2 import NeuralNetwork, tanh
birdPic = pygame.image.load("bluebird-downflap.png")
birdPic1 = pygame.image.load("bluebird-downflap1.png")
birdPic2 = pygame.image.load("bluebird-downflap2.png")
birdPic3 = pygame.image.load("bluebird-downflap3.png")
backGR = pygame.image.load("background-day.png")
pipe = pygame.image.load("pipe-green.png")

pygame.init()

size = width, height = 576, 512
screen = pygame.display.set_mode(size)
pygame.display.toggle_fullscreen()

pygame.display.set_caption("Flappy Bird!")
clock = pygame.time.Clock()
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 170, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
birdColor = (244, 232, 66)

def showScore(score):
    font = pygame.font.SysFont(None, 40)
    text = font.render("score: "+str(score), True, black)
    screen.blit(text,(0,0))

def showGen(gen):
    font = pygame.font.SysFont(None, 25)
    text = font.render("GENERATION: "+str(gen), True, black)
    screen.blit(text,(0,30))

def showHscore(highScore):
    font = pygame.font.SysFont(None, 25)
    text = font.render("HIGH SCORE: "+str(highScore), True, black)
    screen.blit(text,(0,50))


class Obstacle:
    def __init__(self, x, y, w, h, color, sense):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.sense = sense
        if self.sense == 1:
            self.image = pipe
        else:
            self.image = pygame.transform.flip(pipe, False, True)
        self.xSpeed = 9

    def move(self):
        self.x -= self.xSpeed

    def show(self):
        global pipe
        if self.sense == 1:
            screen.blit(self.image,(self.x, self.y))
        else:
            screen.blit(self.image, (self.x, self.y + self.h - 320))


class Bird:
    def __init__(self, x, y, w, h, picture):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.alive = True
        self.gravity = 2
        self.fly = -8
        self.vitesse = 0
        self.score = 0
        self.picture = picture
        self.brain = NeuralNetwork(4, [6], 2, tanh)

    def think(self, whatToSee):
        if self.alive == True:
            choice = self.brain.feed_forward(whatToSee)[-1]
            if choice[1][0] > choice[0][1]:
                self.flying()

    def flying(self):
        self.vitesse = self.fly

    def fall(self, times):
        if times%2 == 0:
            self.vitesse += self.gravity
        self.y += self.vitesse

    def show(self):
        screen.blit(self.picture, (self.x, self.y))
 			


obstaclesUp = []
obstaclesDown = []
to_showUp = []
to_showDown = []
birds = []
void = 120
numBirds = 200
score = 0
timeseu = 0
gen = 1
highScore = 0
lastGeneration = [] 
MutationRate = 0.1
Total = False






def mainGame():
    global obstaclesUp
    global obstaclesDown
    global birds
    global myFont
    global score
    global to_showUp
    global to_showDown
    global void
    global timeseu
    global gen
    global lastGeneration
    global MutationRate
    global Total
    global highScore
    if Total == True:

        if obstaclesUp[-1].x <= 350:
            obstaclesUp.append(Obstacle(width -10, 0, 53, random.randint(100, 300), green, -1))
            obstaclesDown.append(Obstacle(width -10, obstaclesUp[-1].h + void , 53, height - (obstaclesUp[-1].h + void), green, 1))
            to_showUp.append(Obstacle(width -10, 0, 53, obstaclesUp[-1].h, green, -1))
            to_showDown.append(Obstacle(width -10, obstaclesUp[-1].h + void , 53, height - (obstaclesUp[-1].h + void), green, 1))

        for i in range(len(to_showUp)-1, -1, -1):
            to_showUp[i].move()
            to_showDown[i].move()
            to_showUp[i].show()
            to_showDown[i].show()
            if to_showUp[i].x <= -to_showUp[i].w - 10:
                to_showUp.remove(to_showUp[i])
                to_showDown.remove(to_showDown[i])

        for i in range(len(obstaclesUp)-1, -1, -1):
            closetDistance = 10000
            obstaclesUp[i].move()
            obstaclesDown[i].move()
            if obstaclesUp[i].x + obstaclesUp[i].w <= 150:
                score += 1
                obstaclesUp.remove(obstaclesUp[i])
                obstaclesDown.remove(obstaclesDown[i])

        if len(birds) == 0:
            Total = False
            gen += 1

        for bird in birds:
            if len(birds) > 0 :
                whatToSee = np.array([bird.vitesse, obstaclesUp[0].x - bird.x, obstaclesUp[0].y - bird.y, obstaclesDown[0].y - bird.y]).T
                bird.fall(timeseu)
                bird.think(whatToSee)
                bird.show()

                if bird.y <= 0 or bird.y >= height :
                    lastGeneration.append(bird)
                    birds.remove(birds[birds.index(bird)])


                if bird in birds:
                    if obstaclesUp[0].x - bird.w <= bird.x <= obstaclesUp[0].x + obstaclesUp[0].w:
                        if (bird.y <= (obstaclesUp[0].y + obstaclesUp[0].h)) or bird.y >= (obstaclesDown[0].y - bird.h):
                            lastGeneration.append(bird)
                            birds.remove(birds[birds.index(bird)])   

        if highScore < score:
            highScore = score   
        

        timeseu += 1
        showScore(score)
        showGen(gen)
        showHscore(highScore)



Open = True
while Open:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            birds[-1].brain.save()
            Open = False

    screen.blit(backGR, (0, 0))
    screen.blit(backGR, (288, 0))
    if Total == False:
        obstaclesUp = []
        obstaclesDown = []
        to_showUp = []
        to_showDown = []
        void = void
        score = 0
        obstaclesUp.append(Obstacle(width -10, 0, 53, random.randint(100, 300), green, -1))
        obstaclesDown.append(Obstacle(width -10, obstaclesUp[0].h + void , 53, height - (obstaclesUp[0].h + void), green, 1))
        to_showUp.append(Obstacle(width -10, 0, 53, obstaclesUp[0].h, green, -1))
        to_showDown.append(Obstacle(width -10, obstaclesUp[0].h + void , 53, height - (obstaclesUp[0].h + void), green, 1))
        if gen > 1:
            winner = lastGeneration[-1]
            winner.picture = birdPic2
            winner.brain.save()
            lastGeneration = []
            birds = []
            for i in range(numBirds-1):
                birds.append(Bird(150, height/2, 34, 24, birdPic3))
            for i in range(len(birds)):
                birds[i].brain.replace(winner.brain)
                birds[i].brain.mutate(MutationRate)
            birds.append(winner)
        else:
            birds = [Bird(150, height/2, 34, 24, birdPic3) for i in range(numBirds)]
        Total = True
    else:
        mainGame()


    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
