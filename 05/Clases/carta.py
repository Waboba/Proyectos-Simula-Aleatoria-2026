"""
Representación de cartas.
 
Para el motor y el entrenamiento las cartas son ENTEROS del 0 al 51:
 
    id = (numero - 2) * 4 + indice_pinta
 
    numero        : 2..10, Jota=11, Reina=12, Rey=13, As=14
    indice_pinta  : 0=Pica, 1=Trébol, 2=Diamante, 3=Corazón
 
    numero = (id >> 2) + 2        pinta = id & 3
 
(0-51 en vez de 1-52 para poder usar el id directamente como índice de
arreglos numpy / one-hot / tablas Q sin restar 1.)
 
La clase Carta se mantiene solo como "vista" legible (p. ej. para la
parte gráfica): Carta.from_int(id) y carta.to_int() convierten.
"""

#Establecemos constantes numericas y los respectivos indices de cada pinta. (0 a 3)
PINTAS = ("Pica", "Trébol", "Diamante", "Corazón")
INDICE_PINTA = {pinta: i for i, pinta in enumerate(PINTAS)}
FIGURAS = {11: "Jota", 12: "Reina", 13: "Rey", 14: "As"}
 
 
def id_de(numero: int, pinta) -> int:
    """Construye el id (0-51). `pinta` puede ser el nombre o su índice."""
    #Nos preguntamos si la pinta es un string, si es asi, obtenemos su indice en el diccionario de pintas.
    if isinstance(pinta, str):
        pinta = INDICE_PINTA[pinta]
    #Luego, calculamos el id de la carta usando la formula: (numero - 2) * 4 + pinta obteniendo 0-51 ids asociados a cada carta.
    #Notar que la fórmula resta 2 al número de la carta para que el rango de números válidos (2-14) se mapee a un rango de índices (0-12), 
    #y luego multiplica por 4 para obtener el índice base de la pinta correspondiente. Finalmente, se suma el índice de la pinta para obtener el id final.
    return (numero - 2) * 4 + pinta

#Notar que esto es una suerte de clases de equivalencia, donde por ejemplo,
#el id 0 representa el 2 de picas, el id 1 representa el 2 de tréboles, el id 2 representa el 2 de diamantes y el id 3 representa el 2 de corazones.
 
 #Esta funcion lo que hace es obtener el numero de la carta a partir del id de la carta.
 #Para demostrar esta fórmula basta dividr el id y notar que pinta está entre 0 y 3 por lo que su division entera por 4 es 0.
def numero_de(carta_id: int) -> int:
    return (carta_id >> 2) + 2
 
#Lo que hace esta formula es que toma la expresión binaria del id de la carta y utiliza el tres como máscara (que recupera los dos últimos bits)
#Como cualquier multiplo de 4 termina en 00 y la suma de la pinta utiliza a lo más dos bits, esta mascara nos recupera la pinta.
def pinta_de(carta_id: int) -> int:
    return carta_id & 3
 
#Esta función a partir de las auxiliares anteriores recupera, usando el diccionario de figuras, el nombre de la carta a partir del id de la carta.
def nombre_de(carta_id: int) -> str:
    numero = numero_de(carta_id)
    return f"{FIGURAS.get(numero, numero)} de {PINTAS[carta_id & 3]}"
 
 
class Carta:
    #Iniciamos la clase carta con sus atributos: pinta, figura (None para los números) y numero.
    def __init__(self, pinta: str, figura: str | None, numero: int) -> None:
        self.pinta = pinta
        self.figura = figura
        self.numero = numero
    #Herramienta para imprimir la carta de manera legible, si tiene figura se imprime la figura y la pinta, si no se imprime el numero y la pinta.
    def __str__(self) -> str:
        if self.figura is not None:
            return self.figura + " de " + self.pinta
        return str(self.numero) + " de " + self.pinta
    #Representación de la carta como string, para que al imprimir la carta se vea legible.
    def __repr__(self) -> str:
        return self.__str__()
    #Esta función convierte la carta a un entero (id) usando la función id_de.
    def to_int(self) -> int:
        return id_de(self.numero, self.pinta)
 
    @staticmethod
    #Usamos las funciones auxiliares para recuperar la pinta, figura y numero de la carta a partir del id de la carta,
    #luego creamos una instancia de la clase Carta con esos valores.
    def from_int(carta_id: int) -> "Carta":
        numero = numero_de(carta_id)
        return Carta(PINTAS[carta_id & 3], FIGURAS.get(numero), numero)
    #Generamos un diccionario con los atributos de la carta, para poder serializarla y guardarla en un archivo o enviarla por la red.
    def to_dict(self) -> dict:
        return {"pinta": self.pinta, "figura": self.figura, "numero": self.numero}
 
    @staticmethod
    #Dado un diccionario extraemos los valores de pinta, figura y numero y creamos una instancia de la clase Carta con esos valores.
    def from_dict(data: dict) -> "Carta":
        return Carta(pinta=data["pinta"], figura=data["figura"], numero=data["numero"])