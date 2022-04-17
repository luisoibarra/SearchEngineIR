import 'package:ir_search_engine/models/document_model.dart';
import 'package:json_annotation/json_annotation.dart';

part 'query_response_model.g.dart';

@JsonSerializable()
class QueryResponseModel {
  
  final List<DocumentModel> documents;

  QueryResponseModel({required this.documents});

  factory QueryResponseModel.fromJson(Map<String, dynamic> json) => _$QueryResponseModelFromJson(json);

  Map<String, dynamic> toJson() => _$QueryResponseModelToJson(this);

}