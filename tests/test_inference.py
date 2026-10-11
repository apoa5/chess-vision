import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('torch')
pytest.importorskip('torchvision')
import torch
from src.vision.classification import SquarePrediction
from src.vision.inference import PieceClassifier, reconstruct_board, predict_rectified
from src.vision.dataset_loader import ContextSquareDataset, preprocess_context_crop
from src.vision.perspective import RectifiedBoard
from src.vision.dataset import chess_square_at
from src.utils.image_io import save_image


@pytest.mark.parametrize('turns',range(4))
def test_reconstruction_preserves_piece_and_confidence_coordinates(turns):
    canonical=np.zeros((8,8),int); canonical[7,0]=6;canonical[0,7]=12;canonical[3,4]=2
    camera=np.rot90(canonical,-turns)
    confidence=np.arange(64).reshape(8,8)/100+.3
    predictions=[SquarePrediction(r,c,int(camera[r,c]),float(confidence[r,c])) for r in range(8) for c in range(8)]
    result=reconstruct_board(list(reversed(predictions)),turns)
    assert result['board']['rows'][7][0]=='K'
    assert result['board']['rows'][0][7]=='k'
    assert result['board']['rows'][3][4]=='N'
    np.testing.assert_array_equal(result['confidence_matrix'],np.rot90(confidence,turns))
    for p in predictions:
        if p.symbol=='K': assert chess_square_at(p.row,p.col,turns)=='a1'


def test_missing_duplicate_or_unknown_orientation_is_rejected():
    predictions=[SquarePrediction(r,c,0,1.) for r in range(8) for c in range(8)]
    with pytest.raises(ValueError): reconstruct_board(predictions,None)
    with pytest.raises(ValueError): reconstruct_board(predictions[:-1],0)
    with pytest.raises(ValueError): reconstruct_board(predictions[:-1]+[predictions[0]],0)


def test_loader_and_inference_preprocessing_are_identical(tmp_path):
    image=np.random.default_rng(4).integers(0,256,(40,40,3),dtype=np.uint8)
    path=save_image(tmp_path/'crop.png',image)
    polygon=[[10,10],[30,10],[29,31],[9,29]]
    manifest={'split_status':'position_grouped','preview_only':False,'results':[
        {'crop_id':'x','position_id':'p','source_image':'s','split':'test','crop_path':str(path),
         'target_polygon':polygon,'class_id':2}]}
    metadata=tmp_path/'manifest.json';metadata.write_text(json.dumps(manifest))
    actual,_=ContextSquareDataset(metadata,'test',tensor=False)[0]
    expected=preprocess_context_crop(image,polygon)
    np.testing.assert_array_equal(actual,expected)
    assert set(np.unique(actual[3]))=={0.,1.}


def test_classifier_batches_once_per_chunk_without_coordinate_inputs():
    class Model:
        calls=[]
        def __call__(self,batch):
            self.calls.append(batch.shape)
            logits=torch.zeros(len(batch),13);logits[:,2]=5
            return logits
    classifier=PieceClassifier.__new__(PieceClassifier)
    classifier.input_size=32;classifier.batch_size=16;classifier.model=Model()
    crops=[{'image':np.zeros((40,40,3),np.uint8),'target_polygon':[[10,10],[30,10],[30,30],[10,30]],
            'row':r,'col':c} for r in range(8) for c in range(8)]
    result=classifier.predict(crops)
    assert len(result)==64 and all(p.symbol=='N' and len(p.probabilities)==13 for p in result)
    assert classifier.model.calls==[torch.Size([16,4,32,32])]*4


def test_clipped_squares_and_rejected_grid_are_not_silently_predicted():
    image=np.zeros((80,80,3),np.uint8);mask=np.zeros((80,80),np.uint8)
    board=RectifiedBoard(image,mask,(0,0,80,80))
    polygons=[[[c*10,r*10],[c*10+9,r*10],[c*10+9,r*10+9],[c*10,r*10+9]] for r in range(8) for c in range(8)]
    with pytest.raises(ValueError,match='Grid fit rejected'):
        predict_rectified(None,board,{'fit_status':'rejected_seed_preview_only'},0)
    with pytest.raises(ValueError,match='clipped'):
        predict_rectified(None,board,{'fit_status':'fitted','square_polygons':polygons},0)


def test_preprocessing_rejects_outside_polygon():
    with pytest.raises(ValueError,match='within'):
        preprocess_context_crop(np.zeros((40,40,3),np.uint8),[[-1,0],[20,0],[20,20],[0,20]])


def test_cli_fails_explicitly_when_orientation_is_unknown(tmp_path,monkeypatch):
    from scripts import predict_position as cli
    from src.vision.orientation import OrientationResult
    image=save_image(tmp_path/'context.png',np.zeros((80,80,3),np.uint8))
    mask=save_image(tmp_path/'mask.png',np.full((80,80),255,np.uint8))
    geometry=tmp_path/'geometry.json'
    geometry.write_text(json.dumps({'board_bounds':[0,0,80,80],'mask_path':str(mask)}))
    monkeypatch.setattr(cli,'resolve_orientation',lambda *args:OrientationResult('uncertain',reason='Test'))
    monkeypatch.setattr('sys.argv',['predict_position.py',str(image),'--mode','rectified','--geometry',str(geometry),'--output-dir',str(tmp_path/'out')])
    with pytest.raises(SystemExit) as error: cli.main()
    assert error.value.code==1
    status=json.loads((tmp_path/'out'/'context'/'status.json').read_text())
    assert status['status']=='failed' and 'Orientation uncertain' in status['error']


def test_rectified_geometry_does_not_use_ground_truth_fen(tmp_path):
    from scripts.predict_position import load_rectified_geometry
    image=save_image(tmp_path/'context.png',np.zeros((80,80,3),np.uint8))
    mask=save_image(tmp_path/'mask.png',np.full((80,80),255,np.uint8))
    path=tmp_path/'geometry.json'
    path.write_text(json.dumps({'board_bounds':[0,0,80,80],'mask_path':str(mask),'label_rotation':2,
                                'fen':'This is deliberately not a FEN','classes':['invalid']}))
    board,grid,turns,source=load_rectified_geometry(image,path)
    assert turns==2 and grid is None and board.image.shape==(80,80,3)
