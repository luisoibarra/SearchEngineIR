import 'dart:convert';
import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:google_clone/models/document_model.dart';
import 'package:google_clone/models/query_response_model.dart';
import 'package:google_clone/services/api_configuration_service.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

import 'package:dio/dio.dart';

class ApiService {
  bool useDummyData = false;
  static const _getQueryPath = "query";
  static const _getDocumentPath = "document";
  static const _applyFeedbackPath = "feedback";

  Future<QueryResponseModel?> fetchData(
      {required BuildContext context,
      required String query,
      required int offset}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getQueryPath,
          queryParams: {"query": query, "offset": offset.toString()});
      late http.Response response ;
      try {
         response = await http.get(uri);
      } catch (e) {
        log(e.toString());
      }
       
      return QueryResponseModel.fromJson(
          jsonDecode(response.body) as Map<String, dynamic>);
    }
    return QueryResponseModel(documents:[DocumentModel(documentName: "asd", documentDir: "34567")], responseTime: 1);
  }

  Future<String?> fetchDocument(
      {required BuildContext context,
      required String documentDir}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getDocumentPath,
          queryParams: {"document_dir": documentDir});
      final response = await http.get(uri);
      return response.body;
    }
    return null;
  }

  Future<bool> applyFeedback({
    required BuildContext context,
    required String query,
    required List<String> relevantDocumentsDirs,
    required List<String> notRelevantDocumentsDirs
  }) async {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context, listen: false);
      final uri = await apiConfigurationService.getUrl(_applyFeedbackPath);
      final json = {"query": query, "relevants": relevantDocumentsDirs, "not_relevants": notRelevantDocumentsDirs};
      final jsonBody = jsonEncode(json);
      final response = await http.post(uri, headers: {"Content-Type": "application/json"}, body: jsonBody);
      return response.statusCode == 200;
  }
}
