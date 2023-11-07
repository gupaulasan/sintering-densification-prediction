import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import inquirer

ytzp = {'material':['yttria-stabilized zirconia'], 'chemical_reference':['3YTZP'], 'A':[595], 'Ea' :[191500], 'n':[0.4]}
materials = pd.DataFrame.from_dict(ytzp, orient='columns')

def gomez_hotza(Tmax:float, Tinit:float, beta:float, n:float, A: float, Ea: float, theta0: float, G:float, total_time:float, dt:float, R = 8.31446262, beta_SI = False, as_df = False):
    """
    Returns a 2D array with the values of time or temperature and the density or porosity of a material in a sintering process
    
    Args:
    
        Tmax: maximum temperature in the process, in °C
        Tinit: initial temperature of process, in °C
        beta: heating rate of the process. Either in °C(K)/min or °C(K)/s
        n: mass transport coefficient, varies from 0.0 to 1.0
        A: Gomez-Hotza preexponent factor
        Ea: material's activation energy, in J/mol
        theta0: initial porosity, dimensionless
        G: mean particle size, in m
        total_time: the total time in the process, in s
        dt: size of each time step, in s
        R: molar gas constant
        beta_SI: wheter the beta value is expressed in °C(K)/s or not
        as_df: if True, the function will return a pandas DataFrame, if False, the function will return a numpy array.
    """

    #Fixed numbers    
    tsteps = int(total_time/dt)
    Tmax += 273
    Tinit += 273
    
    if not beta_SI:
        betaSI = beta/60
    else:
        betaSI = beta
    
    #Initialize arrays
        #Time arrays
    time = np.arange(0, total_time, dt)
    tn = time**(n)
    tc = time
    
        #Temperature arrays
    T = np.ones(tsteps) *Tmax
    T[0] = Tinit
    n1 = np.zeros(tsteps)
    p = -1
    
    for i in range(tsteps):
        T[i] = Tinit + betaSI*time[i]
        n1[i] = np.exp(-Ea/(R*T[i]))
        p += 1

        if T[i] > Tmax:
            T[i] = Tmax
            break #breaks in case the temperature exceeds Tmax, begining of isothermal regime
    p -= 1
    #Porosity and density arrays
    theta = np.zeros(tsteps)
    rho = np.ones(tsteps)
    theta[0] = theta0
    rho[0] = 1 - theta0
    for i in range(p):
        theta[i] = theta0 * np.exp(-(R/Ea) * ((A*n)/G) *n1[i]* tn[i])
        rho[i] = 1 - theta[i]

    #After reaching Tmax temperature
    for i in range(p, tsteps):
        theta[i] = theta[p-1] * np.exp(-(A/G) * np.exp(-Ea/(R*T[p])) * (tn[i] - tn[p-1])/T[p-1])
        rho[i] = 1-theta[i]
    
    results = np.zeros((4, tsteps))
    results[0,:] = tc
    results[1,:] = T
    results[2,:] = rho
    results[3,:] = theta
    
    if as_df:
        results = pd.DataFrame(results.T, columns=['time','temperatureK', 'density', 'porosity'])
        results['time_min'] = results['time']/60
        results['temperatureC'] = results['temperatureK'] - 273
    
    return results

question1 = [
  inquirer.List(name='material',
                message="Qual material você deseja sinterizar?",
                choices=['3YTZP'],
            ),
]
answer1 = inquirer.prompt(question1)
material = answer1['material']
print('Responda as perguntas abaixos apenas com números\n')
userG = float(input('Qual é o tamanho, em m, de partícula do seu material? '))
userBeta = float(input('Qual taxa de aquecimento você deseja utilizar? '))
question2 = [
  inquirer.List(name='unidade',
                message="Em qual unidade você indicou a taxa de aquecimento?",
                choices=['K/min', 'K/s'],
            ),
]
answer2 = inquirer.prompt(question2)
userUnidade = answer2['unidade']
if userUnidade == 'K/min':
    userBetaSI = False
else:
    userBetaSI = True
userTmax = float(input('Qual é a temperatura máxima, em °C, que você deseja realizar a sinterização? '))
userTinit = float(input('Qual é a temperatura inicial, em °C, que você deseja realizar a sinterização? '))
userTime = float(input('Por quanto tempo, em s, você deseja sinterizar seu material? '))
userTheta0 = float(input('Qual é a porosidade relativa inicial de seu compactado? '))
materialRow = materials[materials['chemical_reference'] == material]
userA = float(materialRow['A'])
userEa = float(materialRow['Ea'])
userN = float(materialRow['n'])
# print(type(userA), type(userEa), type(userN), type(userG), type(userBeta), type(userUnidade), type(userBetaSI), type(userTmax), type(userTinit), type(userTime), type(userTheta0))
print('Simulando o processo...')

resultado = gomez_hotza(Tmax=userTmax,
                        Tinit=userTinit,
                        beta=userBeta,
                        n=userN,
                        A=userA,
                        Ea=userEa,
                        theta0=userTheta0,
                        G=userG,
                        total_time=userTime,
                        dt=5,
                        beta_SI=userBetaSI,
                        as_df=True
                        )

sns.lineplot(data = resultado, 
             x='temperatureC',
             y='density',
             )

plt.xlabel('Temperatura, °C')
plt.ylabel('Densidade relativa')
plt.title(f'Simulação de {material} a {userBeta}{userUnidade}')
plt.show()