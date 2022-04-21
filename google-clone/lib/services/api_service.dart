import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:google_clone/models/query_response_model.dart';
import 'package:google_clone/services/api_configuration_service.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

class ApiService {
  bool useDummyData = false;
  static const _getQueryPath = "query";

  Future<QueryResponseModel?> fetchData(
      {required BuildContext context,
      required String query,
      required int offset}) async {
    if (!this.useDummyData) {
      final apiConfigurationService =
          Provider.of<ApiConfigurationService>(context);
      final uri = await apiConfigurationService.getUrl(_getQueryPath,
          queryParams: {"query": query, "offset": offset.toString()});
      final response = await http.get(uri);
      return QueryResponseModel.fromJson(
          jsonDecode(response.body) as Map<String, dynamic>);
    }
    return null;
  }
}
