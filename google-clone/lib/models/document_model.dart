import 'package:json_annotation/json_annotation.dart';

part 'document_model.g.dart';

@JsonSerializable()
class DocumentModel {

  final String documentName;

  DocumentModel({required this.documentName});

  factory DocumentModel.fromJson(Map<String, dynamic> json) => _$DocumentModelFromJson(json);

  Map<String, dynamic> toJson() => _$DocumentModelToJson(this);

}