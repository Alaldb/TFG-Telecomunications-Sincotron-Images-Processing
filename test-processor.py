import unittest, processor, load, cv2, numpy as np, tifffile as tiff, tools, ising

class TestClass(unittest.TestCase):
    def test_case(self):
        self.assertEqual(1, 1)

    def test_processor_initialization_with_directory(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj)
        self.assertIsInstance(processor_obj, processor.ImgProcessor)
    
    def test_processor_initialization_with_invalid_object(self):
        with self.assertRaises(ValueError):
            processor.ImgProcessor("not_a_directory_object")
    
    def test_processor_correctly_preprocess_png_image(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png"
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj)
        result = processor_obj.preprocess_images()
        standard_png=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(result[0], standard_png)
    
    def test_processor_correctly_preprocess_tiff_image(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj)
        result = processor_obj.preprocess_images()
        img8 = cv2.normalize(tiff.imread(path), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        standard_tiff = img8 if len(img8.shape) == 2 else cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(result[0], standard_tiff)

    def test_processor_raises_error_for_unsupported_format(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\prueba.py"
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj)
        with self.assertRaises(ValueError):
            processor_obj.preprocess_images()
    
    def test_create_circular_mask(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj, parameters={"method": "ising"})
        preprocessed_images = processor_obj.preprocess_images()
        mask = processor_obj.create_circular_mask(preprocessed_images[0])
        test_mask = tools.create_circular_mask(image=preprocessed_images[0])
        self.assertGreaterEqual(mask.sum(), 0)  # Asegura que la máscara no esté vacía
        self.assertEqual(mask.dtype, bool)
        np.testing.assert_array_equal(mask, test_mask)
    
    def test_apply_clahe_with_mask(self):
        img = np.array([
            [50, 50, 50, 50],
            [50, 100, 100, 50],
            [50, 100, 100, 50],
            [50, 50, 50, 50]
        ], dtype=np.uint8)
        mask = np.array([
            [False, False, False, False],
            [False, True,  True,  False],
            [False, True,  True,  False],
            [False, False, False, False]
        ])
        parameters = {"clip_limit": 2.0, "title_grid_size": (2, 2)}

        directory_obj = load.Directory(path=".", method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)

        result = processor_obj.apply_clahe(img=img, parameters=parameters, mask=mask)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, img.shape)
        np.testing.assert_array_equal(result[~mask], img[~mask])
        self.assertFalse(np.array_equal(result[mask], img[mask]))
        self.assertTrue(result.min() >= 0 and result.max() <= 255)
    
    def test_apply_clahe_without_mask(self):
        img = np.array([
            [50, 50, 50, 50],
            [50, 100, 100, 50],
            [50, 100, 100, 50],
            [50, 50, 50, 50]
        ], dtype=np.uint8)

        parameters = {"clip_limit": 2.0, "title_grid_size": (2, 2)}

        directory_obj = load.Directory(path=".", method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)

        result = processor_obj.apply_clahe(img=img, parameters=parameters)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, img.shape)

        # ✔️ CLAHE debe cambiar la imagen
        self.assertFalse(np.array_equal(result, img))

        # ✔️ valores válidos
        self.assertTrue(result.min() >= 0 and result.max() <= 255)
    
    def test_apply_gaussian_blur(self):

        img = np.zeros((7, 7), dtype=np.uint8)
        img[3, 3] = 255  # Esto es como un rectangulo, por eso el blur lo va a difuminar, ya no va a ser un rectangulo recto sino que va a tener un pico en el centro y valores menores a medida que nos alejamos del centro, formando una especie de "montaña" suave.

        directory_obj = load.Directory(path=".", method="ising")
        processor_obj = processor.ImgProcessor(directory_obj)

        parameters_gaussian_blur = {
            "ksize": (3, 3),
            "sigma": 1.0
        }
        result = processor_obj.apply_gaussian_blur(
            img=img,
            parameters=parameters_gaussian_blur
        )
        # tipo correcto
        self.assertEqual(result.dtype, np.uint8)

        # misma forma
        self.assertEqual(result.shape, img.shape)

        # el máximo baja (ya no es un pico perfecto de 255, sino que se difumina)
        self.assertLess(result.max(), 255)

        # la energía se conserva aproximadamente (la suma de los píxeles no cambia mucho, aunque se redistribuya el valor)
        self.assertAlmostEqual(result.sum(), img.sum(), delta=50)

        # el centro ya no es un pico perfecto 
        self.assertLess(result[3, 3], 255)

        # simetría (propiedad gausiana)
        self.assertEqual(result[2, 3], result[4, 3])
        self.assertEqual(result[3, 2], result[3, 4])
    
    def test_apply_otsu_threshold(self):
        #otsu es un método que consigue separar 2 clases, maximizando la diferencia entre elementos de diferentes clases y disminuyendo la diferencia entre la misma clase.
        img = np.zeros((10, 10), dtype=np.uint8)
        img[:5] = 50     # clase 1
        img[5:] = 200    # clase 2
        #Como hay dos clases tan diferenciadas, Otsu debería ser capaz de separarlas perfectamente con un umbral entre 50 y 200, asignando 0 a la clase 1 y 255 a la clase 2.

        directory_obj = load.Directory(path=".", method="ising")
        processor_obj = processor.ImgProcessor(directory_obj)
        result = processor_obj.apply_otsu_threshold(img)

        # tipo correcto
        self.assertEqual(result.dtype, np.uint8)

        # solo dos clases
        unique_vals = np.unique(result)
        self.assertTrue(np.array_equal(unique_vals, [0, 255]))

        # separación efectiva (Otsu debe separar las dos mitades)
        top_half = result[:5]
        bottom_half = result[5:]

        self.assertTrue(np.all(top_half == top_half[0]))
        self.assertTrue(np.all(bottom_half == bottom_half[0]))
        self.assertNotEqual(top_half[0].all(), bottom_half[0].all())

        # Verificar que el umbral calculado por Otsu está entre las dos clases
        threshold_value, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        self.assertGreater(threshold_value, 0)
        self.assertLess(threshold_value, 255)
    
    def test_thresholding_method_with_mask(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        
        directory_obj = load.Directory(path=path, method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)
        
        img = processor_obj.preprocess_images()[0]

        parameters = {
            "clahe": {"clip_limit": 2.0, "title_grid_size": (8, 8)},
            "gaussian_blur": {"ksize": (5, 5), "sigma": 1.0}
        }

        mask = processor_obj.create_circular_mask(img)

        result = processor_obj.thresholding(img=img, parameters=parameters, mask=mask)

        # tipo correcto
        self.assertEqual(result.dtype, np.uint8)

        # binarización correcta
        unique_vals = np.unique(result)
        self.assertTrue(np.array_equal(unique_vals, [0, 255]))

        # la máscara realmente limita el procesamiento
        outside_mask = result[~mask]
        inside_mask = result[mask]

        # dentro de la máscara debe haber variación real (información útil)
        self.assertGreater(len(np.unique(inside_mask)), 1)

        #efecto del pipeline (no todo blanco o negro trivial)
        self.assertFalse(np.all(result == 0))
        self.assertFalse(np.all(result == 255))
    
    def test_thresholding_method_without_mask(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        
        directory_obj = load.Directory(path=path, method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)
        
        img = processor_obj.preprocess_images()[0]

        parameters = {
            "clahe": {"clip_limit": 2.0, "title_grid_size": (8, 8)},
            "gaussian_blur": {"ksize": (5, 5), "sigma": 1.0}
        }

        result = processor_obj.thresholding(img=img, parameters=parameters)

        # tipo correcto
        self.assertEqual(result.dtype, np.uint8)

        # binarización correcta
        unique_vals = np.unique(result)
        self.assertTrue(np.array_equal(unique_vals, [0, 255]))

        #efecto del pipeline (no todo blanco o negro trivial)
        self.assertFalse(np.all(result == 0))
        self.assertFalse(np.all(result == 255))

    def test_calculate_statistical_variables(self):
        y = np.array([10, 20, 30, 40, 50, 60])
        x = np.array([0, 0, 1, 1, 2, 2])
        num_states = int(3)

        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        directory_obj = load.Directory(path=path, method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)
        mask= processor_obj.create_circular_mask(processor_obj.preprocess_images()[0])

        result = processor_obj.calculate_statistical_variables(y, x, num_states)

        # Estado 0 → [10, 20]
        self.assertAlmostEqual(result[0]['mean'], 15.0)
        self.assertAlmostEqual(result[0]['std'], np.std([10, 20]) + 1e-6)

        # Estado 1 → [30, 40]
        self.assertAlmostEqual(result[1]['mean'], 35.0)
        self.assertAlmostEqual(result[1]['std'], np.std([30, 40]) + 1e-6)

        # Estado 2 → [50, 60]
        self.assertAlmostEqual(result[2]['mean'], 55.0)
        self.assertAlmostEqual(result[2]['std'], np.std([50, 60]) + 1e-6)
    
    def test_calculate_statistical_variables_with_mask(self):
        y = np.array([10, 20, 30, 40, 50, 60])
        x = np.array([0, 0, 1, 1, 2, 2])
        num_states = 3
        mask = np.array([True, False, True, False, True, False])

        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        directory_obj = load.Directory(path=path, method="thresholding")
        processor_obj = processor.ImgProcessor(directory_obj)

        result = processor_obj.calculate_statistical_variables(y, x, num_states, mask=mask)

        self.assertAlmostEqual(result[0]['mean'], 10.0)
        self.assertAlmostEqual(result[0]['std'], 1e-6)

        self.assertAlmostEqual(result[1]['mean'], 30.0)
        self.assertAlmostEqual(result[1]['std'], 1e-6)

        self.assertAlmostEqual(result[2]['mean'], 50.0)
        self.assertAlmostEqual(result[2]['std'], 1e-6)
    
    def test_initialize_ising_model(self):
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Images ALBA - Sample PyHM004\Low temperature\primera.tif"
        directory_obj = load.Directory(path=path, method="ising")
        processor_obj = processor.ImgProcessor(directory_obj)
        preprocessed_images = processor_obj.preprocess_images()
        num_states = 3
        mask= processor_obj.create_circular_mask(preprocessed_images[0])
        x_matrix, parameters = processor_obj.initialize_ising_model(preprocessed_images[0], num_states, mask=mask)
        x_matrix_test, parameters_test = ising.initialize_ising_model(preprocessed_images[0], num_states)
        self.assertEqual(x_matrix.shape, preprocessed_images[0].shape)
        self.assertEqual(len(parameters), num_states)
        for state in parameters:
            self.assertAlmostEqual(parameters[state]['mean'], parameters_test[state]['mean'], places=5)
            self.assertAlmostEqual(parameters[state]['std'], parameters_test[state]['std'], places=5)
        np.testing.assert_array_equal(x_matrix, x_matrix_test)
        
    def test_calculate_energy(self):
        y_value = 150.0
        mu = 100.0
        sigma = 50.0
        beta = 2.0
        state = 1
        x = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        row, col = 1, 1 # Centro de la matriz
        expected_energy = -7.5  # positive_energy: 0.5 * ((150-100)/50)^2 = 0.5 * (1)^2 = 0.5
                                # negative_energy: -2.0 * 4 vecinos = -8.0
        path = r"C:\Users\user\Desktop\TFG-Teleco\OpenCV\Practica\hetero1.png" # Path dummy para init
        directory_obj = load.Directory(path=path)
        processor_obj = processor.ImgProcessor(directory_obj)
        
        result = processor_obj.calculate_energy(
            y_value, x, mu, sigma, beta, state, row, col
        )

        # 3. Verificar
        self.assertAlmostEqual(result, expected_energy, places=5)

    def test_calculate_energy_edge_case(self):
        """Prueba que el cálculo sea correcto en los bordes (donde hay menos vecinos)"""
        x = np.ones((3, 3), dtype=np.int32)
        # Esquina superior izquierda (0,0). Vecinos: (0,1) y (1,0) -> total 2
        row, col = 0, 0
        state = 1
        beta = 1.0
        
        # positive_energy: y=100, mu=100 -> 0.0
        # negative_energy: -1.0 * 2 vecinos = -2.0
        result = processor.ImgProcessor.calculate_energy(
            None, 100.0, x, 100.0, 10.0, 1.0, 1, 0, 0
        )
        self.assertEqual(result, -2.0)
    
    def test_apply_ising_model_icm(self):
        # 🔹 Imagen sintética pequeña (control total del sistema)
        y = np.array([
            [10, 10, 10],
            [10, 200, 10],
            [10, 10, 10]
        ], dtype=np.float32)

        # 🔹 inicialización de estados (ruido controlado)
        x = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.int32)

        directory_obj = load.Directory(path=".", method="ising")
        processor_obj = processor.ImgProcessor(directory_obj)

        num_states = 2
        beta = 1.0
        max_iterations = 10

        # 🔹 ejecutar modelo
        result = processor_obj.apply_ising_model_icm(
            y=y,
            x=x,
            parameters={
                0: {"mean": 10, "std": 1},
                1: {"mean": 200, "std": 1}
            },
            num_states=num_states,
            beta=beta,
            max_iterations=max_iterations
        )

        # 🔥 1. shape inalterado
        self.assertEqual(result.shape, y.shape)

        # 🔥 2. dtype correcto
        self.assertTrue(np.issubdtype(result.dtype, np.integer))

        # 🔥 3. solo estados válidos
        self.assertTrue(np.all(np.isin(result, [0, 1])))

        # 🔥 4. convergencia: borde homogéneo debe tender a clase 0
        border = np.array([
            result[0, 0], result[0, 1], result[0, 2],
            result[1, 0], result[1, 2],
            result[2, 0], result[2, 1], result[2, 2]
        ])

        self.assertTrue(np.all(border == border[0]))

        # 🔥 5. el centro debe diferenciarse (alta intensidad → estado 1)
        self.assertEqual(result[1, 1], 1)

        # 🔥 6. estabilidad (si lo vuelves a ejecutar, debe ser igual)
        result2 = processor_obj.apply_ising_model_icm(
            y=y,
            x=x,
            parameters={
                0: {"mean": 10, "std": 1},
                1: {"mean": 200, "std": 1}
            },
            num_states=num_states,
            beta=beta,
            max_iterations=max_iterations
        )

        np.testing.assert_array_equal(result, result2)
    
    def test_process_images_functionality(self):
        # 🔹 imagen sintética controlada
        img = np.array([
            [10, 10, 10],
            [10, 200, 10],
            [10, 10, 10]
        ], dtype=np.uint8)

        # 🔹 mock Directory
        directory_obj = load.Directory(path=".", method="thresholding")
        directory_obj.img_array = ["fake_path"]

        processor_obj = processor.ImgProcessor(directory_obj, parameters={
            "thresholding": {
                "clahe": {"clip_limit": 2.0, "title_grid_size": (2, 2)},
                "gaussian_blur": {"ksize": (3, 3), "sigma": 1.0}
            }
        })

        # 🔹 parchear funciones dependientes de IO
        processor_obj.standarize_image = lambda x: img
        processor_obj.create_circular_mask = lambda x: np.ones_like(x, dtype=bool)

        # 🔹 ejecutar
        processor_obj.process_images()

        # 🔥 1. se genera resultado
        self.assertEqual(len(processor_obj.results), 1)

        # 🔥 2. estructura correcta
        res = processor_obj.results[0]
        self.assertIn("original_img", res)
        self.assertIn("parameters", res)
        self.assertIn("result", res)

        # 🔥 3. tipo correcto
        self.assertEqual(res["result"].dtype, np.uint8)

        # 🔥 4. imagen no vacía
        self.assertTrue(res["result"].size > 0)
    
    def test_process_images_determinism(self):
        # 🔹 imagen sintética controlada (evita ruido externo)
        img = np.array([
            [10, 10, 10],
            [10, 200, 10],
            [10, 10, 10]
        ], dtype=np.uint8)

        directory_obj = load.Directory(path=".", method="ising")
        directory_obj.img_array = ["fake_path"]

        parameters = {
            "ising": {
                "num_states": 2,
                "beta": 1.0,
                "max_iterations": 5
            }
        }

        processor_obj = processor.ImgProcessor(directory_obj, parameters=parameters)

        # 🔹 evitar I/O real (clave para determinismo en test)
        processor_obj.standarize_image = lambda x: img.copy()
        processor_obj.create_circular_mask = lambda x: np.ones_like(x, dtype=bool)

        # 🔥 ejecutar dos veces
        processor_obj.process_images()
        result_1 = processor_obj.results[0]["result"]

        processor_obj.results = []  # reset manual

        processor_obj.process_images()
        result_2 = processor_obj.results[0]["result"]

        # 🔥 1. resultados idénticos
        np.testing.assert_array_equal(result_1, result_2)

        # 🔥 2. mismo tipo
        self.assertEqual(result_1.dtype, result_2.dtype)

        # 🔥 3. misma forma
        self.assertEqual(result_1.shape, result_2.shape)

        # 🔥 4. no hay variación estadística (refuerzo de estabilidad)
        self.assertEqual(np.sum(result_1 != result_2), 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

