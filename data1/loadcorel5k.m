function [annottest, annottrain, captiontest, captiontrain, sifttest, sifttrain] = loadcorel5k(pathToData)
    captiontest={};
    captiontrain={};
    ds=pathToData;
    dict=textread([ds '_dictionary.txt'],'%s');
    sets={'test','train'};
    strtest=sets{1};
    strtrain=sets{2};
    listtest=textread([ds '_' strtest '_list.txt'],'%s');
    listtrain=textread([ds '_' strtrain '_list.txt'],'%s');
    annottest=logical(vec_read([ds '_' strtest '_annot.hvecs']));
    annottrain=logical(vec_read([ds '_' strtrain '_annot.hvecs']));
    sifttest=vec_read([ds '_' strtest '_DenseSift.hvecs']);
    sifttrain=vec_read([ds '_' strtrain '_DenseSift.hvecs']);
    utest=randperm(length(listtest));
    utrain=randperm(length(listtrain));
    for i=1:length(listtest)
        %n=utest(i);
        words=dict(annottest(i,:));
        captiontest{i}=words;
    end
    for i=1:length(listtrain)
        %n=utest(i);
        words=dict(annottrain(i,:));
        captiontrain{i}=words;
    end
end


