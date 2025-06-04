#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

class Perceptron {
private:
    std::vector<double> pesos;           // Pesos sinápticos
    double tasa_aprendizaje;             // Tasa de aprendizaje
    double sesgo;                        // Término de sesgo
    int epocas_entrenadas;               // Número de épocas entrenadas
    std::vector<double> errores_por_epoca; // Registro de errores por época

    // Función de activación: escalón
    int funcion_activacion(double suma) {
        return (suma >= 0.0) ? 1 : 0;
    }

public:
    // Constructor
    Perceptron(int num_entradas, double tasa_aprendizaje = 0.1)
        : tasa_aprendizaje(tasa_aprendizaje), sesgo(0.0), epocas_entrenadas(0) {
        pesos.resize(num_entradas);
        std::srand(static_cast<unsigned int>(std::time(0)));
        for (auto& peso : pesos) {
            peso = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    // Método de predicción
    int predecir(const std::vector<double>& entradas) {
        double suma = sesgo;
        for (size_t i = 0; i < entradas.size(); ++i) {
            suma += entradas[i] * pesos[i];
        }
        return funcion_activacion(suma);
    }

    // Método de entrenamiento
    void entrenar(const std::vector<std::vector<double>>& datos_entrenamiento,
                  const std::vector<int>& etiquetas, int epocas) {
        for (int e = 0; e < epocas; ++e) {
            int errores = 0;
            for (size_t i = 0; i < datos_entrenamiento.size(); ++i) {
                int salida = predecir(datos_entrenamiento[i]);
                int error = etiquetas[i] - salida;
                if (error != 0) {
                    ++errores;
                    for (size_t j = 0; j < pesos.size(); ++j) {
                        pesos[j] += tasa_aprendizaje * error * datos_entrenamiento[i][j];
                    }
                    sesgo += tasa_aprendizaje * error;
                }
            }
            errores_por_epoca.push_back(static_cast<double>(errores));
            ++epocas_entrenadas;
            if (errores == 0) {
                break; // Convergencia alcanzada
            }
        }
    }

    // Método para obtener los pesos actuales
    std::vector<double> obtener_pesos() const {
        return pesos;
    }

    // Método para obtener el sesgo actual
    double obtener_sesgo() const {
        return sesgo;
    }

    // Método para obtener el número de épocas entrenadas
    int obtener_epocas_entrenadas() const {
        return epocas_entrenadas;
    }

    // Método para obtener el registro de errores por época
    std::vector<double> obtener_errores_por_epoca() const {
        return errores_por_epoca;
    }
};

int main() {
    // Datos para la compuerta lógica AND
    std::vector<std::vector<double>> entradas = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<int> etiquetas = {0, 0, 0, 1};

    Perceptron p(2);
    p.entrenar(entradas, etiquetas, 10);

    std::cout << "Pruebas del perceptrón entrenado:\n";
    for (const auto& entrada : entradas) {
        std::cout << entrada[0] << " AND " << entrada[1] << " = " << p.predecir(entrada) << "\n";
    }

    // Mostrar información adicional
    std::cout << "\nPesos finales:\n";
    auto pesos = p.obtener_pesos();
    for (size_t i = 0; i < pesos.size(); ++i) {
        std::cout << "Peso " << i << ": " << pesos[i] << "\n";
    }
    std::cout << "Sesgo: " << p.obtener_sesgo() << "\n";
    std::cout << "Épocas entrenadas: " << p.obtener_epocas_entrenadas() << "\n";

    std::cout << "\nErrores por época:\n";
    auto errores = p.obtener_errores_por_epoca();
    for (size_t i = 0; i < errores.size(); ++i) {
        std::cout << "Época " << i + 1 << ": " << errores[i] << " errores\n";
    }

    return 0;
}
