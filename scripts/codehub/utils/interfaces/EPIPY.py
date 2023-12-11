import os
import dearpygui.dearpygui as dpg

# Local imports
import pipeline_manager as PM

# Interface imports
from EPIPY_modules.theme import applyTheme
from EPIPY_modules.pipelines import showPipelines
from EPIPY_modules.configuration import showConfiguration

"""
    A classe Interface é responsável por gerar os elementos visuais do programa.
    O uso da biblioteca DearPyGUI faz com que seja possível executar o programa em windows ou linux
    e ainda utilizar dos benefícios da acerelação de hardware.        
"""
class Interface:

    """
        Define os parâmetros e inicia a função Show.
    """
    def __init__(self,args,metadata):
        self.args = args
        self.help = metadata
        self.show()
        pass

    """
        Cria o contexto e a janela do DPG e invoca a função showTabBar para a renderização de cada uma das tabs e seus conteúdos.
    """
    def show(self):
        dpg.create_context()
        dpg.create_viewport(title='EPIPY: Epilepsy Processing and Interpretation using PYthon', width=1280, height=720, min_height=600, min_width=900)

        with dpg.window(tag="Main"):
            applyTheme()
            self.showTabBar()
            pass
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        pass

    """
        Responsável pela invocação das tabs individuais.
    """
    def showTabBar(self):
        with dpg.tab_bar():
            self.showTabs()
        pass

    """
        Cria as diferentes tabs do programa e chama o método show<Tab> para popular cada uma das abas.
        Processing: Importação e Cropping da imagem.
        Filtering: Ajustes em níveis de cor, brilho, contraste e blurring na imagem.
        Thresholding: Ajustes na binarização da imagem. Para que a binarização ocorra a imagem é automaticamente convertida para tons de cinza.
        Contour Extraction: Extrai o contorno dos objetos presentes na imagem binarizada. Permite a exportação do arquivo .txt com os dados do contorno.
        Mesh Generation: Gera a malha e possibilita as configurações necessária para método matemáticos com os pontos resultantes da aba anterior. Permite a importação de novos pontos.
    """
    def showTabs(self):
        dpg.add_texture_registry(show=False, tag='textureRegistry')
        with dpg.tab(label='Configurations'):
            showConfiguration()
            pass
        with dpg.tab(label='Pipelines'):
            showPipelines()
            pass
        with dpg.tab(label='Channel Settings'):
            #showFiltering(self.callbacks)
            pass
        with dpg.tab(label='Acceptance Criteria'):
            #showThresholding(self.callbacks)
            pass
        with dpg.tab(label='Preprocessing Options'):
            #showContourExtraction(self.callbacks)
            pass
        with dpg.tab(label='Feature Extraction'):
            #showInterpolation(self.callbacks)
            pass
        #self.callbacks.imageProcessing.disableAllTags()
        pass

class App:
    def __init__(self,args,metadata):
        self.interface = Interface(args,metadata)
        pass


if __name__ == '__main__':
    
    # Get the pathing to the pipeline manager. Allows us to find the argument file
    pipeline_path = os.path.dirname(os.path.abspath(PM.__file__))+'/'

    # Get the arguments
    args, metadata = PM.argument_handler(argument_dir=pipeline_path,require_flag=False)
    args           = vars(args)

    # Run the dearpygui app
    app = App(args,metadata)