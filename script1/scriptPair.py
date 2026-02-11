# 1. On demande la saisie (ici pas de int() car on veut du texte)
chaine = input("Entrez votre texte : ")

# 2. On prépare une boîte pour compter, elle démarre à 0
compteur = 0

# 3. On examine chaque lettre de la chaine une par une
for lettre in chaine:
    if lettre == 'e':
        # Si on trouve un 'e', on rajoute 1 au compteur
        compteur = compteur + 1

# 4. On affiche le résultat de façon différenciée
if compteur == 0:
    print("0 : aucun e saisie")
else:
    print(str(compteur) + " : vous avez saisi " + str(compteur) + " e")